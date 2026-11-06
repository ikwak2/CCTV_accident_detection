"""Integration checks for exclusive claims, dependencies, and crash recovery."""
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest

import importlib.util
spec=importlib.util.spec_from_file_location('orv_queue',Path(__file__).with_name('experiment_queue.py'))
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def claim_many(root,name,output):
    db=q.connect(root)
    while True:
        job=q.claim(db,name)
        if job is None:break
        output.put(job['id'])
        db.execute("UPDATE jobs SET state='completed' WHERE id=?",(job['id'],))


class QueueTests(unittest.TestCase):
    def test_two_workers_never_claim_the_same_job(self):
        with tempfile.TemporaryDirectory() as root:
            db=q.connect(root)
            for i in range(100):q.add_job(db,f'job{i}',i,[])
            db.close()  # Never inherit a live SQLite connection across fork.
            output=multiprocessing.Queue()
            processes=[multiprocessing.Process(target=claim_many,args=(root,str(i),output)) for i in range(2)]
            for p in processes:p.start()
            items=[output.get(timeout=10) for _ in range(100)]
            for p in processes:p.join(5);self.assertEqual(p.exitcode,0)
            self.assertEqual(len(set(items)),100)
            db=q.connect(root)
            self.assertEqual(db.execute("SELECT count(*) FROM jobs WHERE state='completed'").fetchone()[0],100)

    def test_failed_dependency_blocks_child_and_stale_lease_recovers(self):
        with tempfile.TemporaryDirectory() as root:
            db=q.connect(root)
            q.add_job(db,'parent',0,[],state='failed')
            q.add_job(db,'child',1,[],['parent'])
            self.assertIsNone(q.claim(db,'worker'))
            db.execute("UPDATE jobs SET state='running',worker='dead',heartbeat=? WHERE id='parent'",(time.time()-121,))
            self.assertIsNone(q.claim(db,'new'))
            q.recover_worker(db,'dead')
            self.assertEqual(q.claim(db,'new')['id'],'parent')
            self.assertIsNone(q.claim(db,'other'))
            db.execute("UPDATE jobs SET state='completed' WHERE id='parent'")
            self.assertEqual(q.claim(db,'other')['id'],'child')


if __name__=='__main__':unittest.main()
