"""Regressions for a late corrupt window and a shared, immutable eval subset."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

from audit_evaluation import audit, evaluation_rows
from common import read_csv, sha256, write_csv
from analyze import bootstrap_pair
from experiment_queue import add_job, connect
from recover_evaluations import recover
from run import score_rows


class EvaluationAuditTests(unittest.TestCase):
    def test_late_16_frame_failure_excludes_video_for_every_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifests').mkdir()
            rows=[dict(id=name,path=name,dataset='UTIC',label='0',split=split,
                       duration='20',fps='10',file_sha256=name)
                  for name,split in [('train','train'),('dev','dev'),('broken','test'),('good','test'),('ucf','external')]]
            source=root/'manifests/grouped.csv'
            write_csv(source,rows)
            original_hash=sha256(source)
            calls=[]
            def decode(path,start,span,count,fps):
                calls.append((path,start,count))
                if path=='broken' and start==16 and count==16:
                    raise RuntimeError('corrupt tail packet')
                return [None]*count,list(range(count))
            with patch('audit_evaluation.read_window',side_effect=decode):
                report=audit(root,workers=2,executor='thread')
            kept,loaded=evaluation_rows(root)
            self.assertEqual([r['id'] for r in kept],['dev','good','ucf'])
            self.assertEqual(report['total_excluded'],1)
            self.assertEqual(report['counts_excluded'],{'test:UTIC:0':1})
            self.assertEqual(sha256(source),original_hash)
            self.assertEqual(report,loaded)
            self.assertFalse(any(p=='train' for p,_,_ in calls))
            self.assertTrue(any(p=='good' and n==32 for p,_,n in calls))
            self.assertEqual(json.loads((root/'manifests/evaluation_exclusions_private.json').read_text())[0]['id'],'broken')
            # An unchanged audit can resume without decoding any completed input.
            with patch('audit_evaluation.read_window',side_effect=AssertionError('unexpected rescan')):
                audit(root,workers=2,executor='thread')
            (root/'manifests/evaluation.csv').write_text('modified')
            with self.assertRaisesRegex(RuntimeError,'evaluation manifest changed'):
                evaluation_rows(root)

    def test_missing_audit_blocks_evaluation_and_mismatched_sets_block_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                evaluation_rows(root)
        with self.assertRaisesRegex(ValueError,'identical evaluated video sets'):
            bootstrap_pair([{'id':'one'}],[{'id':'two'}])

    def test_spawned_decoder_process_reads_real_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifests').mkdir()
            path=root/'clip.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10.,(64,48))
            self.assertTrue(writer.isOpened())
            for i in range(10):writer.write(np.full((48,64,3),i*10,dtype=np.uint8))
            writer.release()
            write_csv(root/'manifests/grouped.csv',[dict(id='clip',path=str(path),dataset='CADP',label='1',
                      split='test',duration='1',fps='10',file_sha256=sha256(path))])
            report=audit(root,workers=1,executor='process')
            self.assertEqual(report['total_eligible'],1)
            self.assertEqual(report['total_excluded'],0)
            self.assertEqual(report['executor'],'process')
            self.assertEqual(evaluation_rows(root)[0][0]['id'],'clip')

    def test_recovery_keeps_completed_training_and_scores_and_backs_up_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            db=connect(root)
            add_job(db,'prepare',0,[],state='completed')
            add_job(db,'train_done',20,[],state='completed')
            add_job(db,'train_active',80,[],state='running')
            add_job(db,'eval_done',25,[],['train_done'],state='failed')
            db.execute("UPDATE jobs SET attempts=2,error='old decoder failure' WHERE id='eval_done'")
            weight=root/'best.pt'
            weight.write_bytes(b'unchanged checkpoint')
            journal=root/'predictions_private.jsonl'
            journal.write_text('cached scores')
            report={'evaluation_manifest_sha256':'eval-hash','source_manifest_sha256':'train-hash'}
            with self.assertRaisesRegex(RuntimeError,'Stop both'):
                recover(root)
            with patch('recover_evaluations.evaluation_rows',return_value=([],report)):
                result=recover(root,workers_stopped=True)
            states={r['id']:dict(r) for r in db.execute('SELECT * FROM jobs')}
            self.assertEqual(states['train_done']['state'],'completed')
            self.assertEqual(states['train_active']['state'],'pending')
            self.assertEqual(states['eval_done']['state'],'pending')
            self.assertEqual(states['eval_done']['attempts'],0)
            self.assertIn('evaluation_audit',json.loads(states['eval_done']['dependencies']))
            old=json.loads((Path(result['backup_directory'])/'jobs_before.json').read_text())
            self.assertEqual(next(r for r in old if r['id']=='eval_done')['error'],'old decoder failure')
            self.assertEqual(weight.read_bytes(),b'unchanged checkpoint')
            self.assertEqual(journal.read_text(),'cached scores')

    def test_cached_timestamps_are_verified_and_excluded_ids_do_not_enter_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            row=dict(id='good',dataset='UTIC',category='normal',label='0',group='camera',
                     split='test',quarantined='0',duration='20',fps='10')
            cached=dict(row,score=.2,label=0,quarantined=0,
                        sample_timestamps=[np.linspace(start,start+3.9,16).tolist()
                                           for start in np.linspace(0,16,5)])
            journal=out/'predictions_private.jsonl'
            journal.write_text(json.dumps(cached)+'\n'+json.dumps(dict(cached,id='excluded'))+'\n')
            with patch('run.read_window',side_effect=AssertionError('cached video decoded again')):
                result=score_rows(Mock(),None,'videomae',[row],out)
            self.assertEqual([r['id'] for r in result],['good'])
            self.assertEqual([r['id'] for r in read_csv(out/'predictions.csv')],['good'])
            self.assertEqual(json.loads((out/'progress.json').read_text())['scored'],1)
            cached['sample_timestamps'][0][0]=999
            journal.write_text(json.dumps(cached)+'\n')
            with self.assertRaisesRegex(RuntimeError,'Cached prediction does not match'):
                score_rows(Mock(),None,'videomae',[row],out)


if __name__=='__main__':
    unittest.main()
