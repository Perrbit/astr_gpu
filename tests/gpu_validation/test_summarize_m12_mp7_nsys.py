from pathlib import Path
import sqlite3
import tempfile
import unittest

from summarize_m12_mp7_nsys import analyze


class M12TraceTest(unittest.TestCase):
    def fixture(self,path):
        db = sqlite3.connect(path)
        db.executescript('''
          CREATE TABLE StringIds(id INTEGER,value TEXT);
          CREATE TABLE MPI_RANKS(globalTid INTEGER,rank INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,
            shortName INTEGER,globalPid INTEGER,registersPerThread INTEGER,localMemoryPerThread INTEGER);
          CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,label TEXT);
          CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,
            globalPid INTEGER,copyKind INTEGER,bytes INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,
            globalTid INTEGER,nameId INTEGER);
          CREATE TABLE MPI_P2P_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,
            textId INTEGER,remoteRank INTEGER,tag INTEGER,size INTEGER);
        ''')
        pid,tid = 123 << 24,(123 << 24)+123
        db.executemany('INSERT INTO StringIds VALUES(?,?)',[
            (1,'mp_positivity_gpu_mp_faces_kernel_'),(2,'cudaDeviceSynchronize'),
            (3,'MPI_Waitall'),(4,'MPI_Isend')])
        db.execute('INSERT INTO MPI_RANKS VALUES(?,0)',(tid,))
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,1,?,128,0)',
                       [(10*i,10*i+6,pid) for i in range(72)])
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,2)',
                       [(90,100,tid),(92,99,tid)])
        db.executemany('INSERT INTO MPI_P2P_EVENTS VALUES(?,?,?,?,?,?,?)',[
            (96,99,tid,3,None,None,None),(106,108,tid,4,1,21401,100)])
        db.commit()
        db.close()

    def test_aligned_cycles_and_interval_union_not_api_sum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'trace.sqlite'
            self.fixture(path)
            row = analyze(path)['ranks']['0']
            self.assertAlmostEqual(row['window_seconds'],540e-9)
            self.assertAlmostEqual(row['kernel_seconds'],324e-9)
            self.assertAlmostEqual(row['sync_without_device_work_seconds'],4e-9)
            self.assertAlmostEqual(row['exposed_mpi_seconds'],5e-9)
            self.assertEqual(row['p2p_metadata'],{'MPI_Isend/tag21401':dict(calls=1,metadata_bytes=100)})

    def test_incomplete_face_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'trace.sqlite'
            self.fixture(path)
            with sqlite3.connect(path) as db:
                db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start=710')
            with self.assertRaisesRegex(ValueError,'8 steps'):
                analyze(path)


if __name__ == '__main__':
    unittest.main()
