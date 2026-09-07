import sqlite3
import unittest

from summarize_mpi_kernel_overlap import summarize, summarize_completion


class MpiKernelOverlapTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1, 'MPI_Testall');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER);
            CREATE TABLE MPI_START_WAIT_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, textId INTEGER);
        ''')

    def test_request_rows_are_deduplicated_and_processes_are_separate(self):
        event = (100, 200, (7 << 24) + 19, 1)
        self.db.executemany('INSERT INTO MPI_START_WAIT_EVENTS VALUES(?,?,?,?)', [event] * 4)
        self.db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)',
                            [(150, 250, 7 << 24), (100, 200, 8 << 24)])
        result = summarize(self.db)['MPI_Testall']
        self.assertEqual(result['unique_calls'], 1)
        self.assertEqual(result['calls_overlapping_kernel'], 1)
        self.assertEqual(result['start_wait_calls_overlapping_kernel'], 1)
        self.assertEqual(result['overlap_interval_sum_ns'], 50)

    def test_overlapping_kernels_do_not_double_count_api_time(self):
        self.db.execute('INSERT INTO MPI_START_WAIT_EVENTS VALUES(100,200,?,1)', (7 << 24,))
        self.db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)',
                            [(110, 180, 7 << 24), (150, 190, 7 << 24)])
        self.assertEqual(summarize(self.db)['MPI_Testall']['overlap_interval_sum_ns'], 80)

    def test_missing_optional_start_wait_table_is_supported(self):
        self.db.execute('DROP TABLE MPI_START_WAIT_EVENTS')
        self.assertEqual(summarize(self.db), {})

    def test_other_table_null_request_calls_are_included_without_duplicates(self):
        self.db.execute('CREATE TABLE MPI_OTHER_EVENTS AS SELECT * FROM MPI_START_WAIT_EVENTS')
        event = (100, 200, 7 << 24, 1)
        for table in ('MPI_OTHER_EVENTS', 'MPI_START_WAIT_EVENTS'):
            self.db.execute(f'INSERT INTO {table} VALUES(?,?,?,?)', event)
        result = summarize(self.db)['MPI_Testall']
        self.assertEqual(result['unique_calls'], 1)
        self.assertEqual(result['calls_overlapping_kernel'], 0)

    def test_missing_cuda_trace_is_an_error(self):
        self.db.execute('DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL')
        with self.assertRaises(ValueError):
            summarize(self.db)

    def test_unattributed_other_calls_do_not_prove_request_overlap(self):
        self.db.execute('CREATE TABLE MPI_OTHER_EVENTS AS SELECT * FROM MPI_START_WAIT_EVENTS')
        self.db.execute('INSERT INTO MPI_OTHER_EVENTS VALUES(100,200,?,1)', (7 << 24,))
        self.db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,200,?)', (7 << 24,))
        result = summarize(self.db)['MPI_Testall']
        self.assertEqual(result['calls_overlapping_kernel'], 1)
        self.assertEqual(result['start_wait_calls'], 0)
        self.assertEqual(result['start_wait_calls_overlapping_kernel'], 0)


class MpiCompletionTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1, 'diffusion_rhs');
            INSERT INTO StringIds VALUES(2, 'ASTR_MPI_TESTALL_COMPLETE');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, globalPid INTEGER, demangledName INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
        ''')

    def marker(self, time, process=7, text='ASTR_MPI_TESTALL_PENDING', text_id=None):
        self.db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)',
                        (time, (process << 24) + 5, text, text_id))

    def test_pending_and_complete_are_distinct_and_support_string_ids(self):
        self.db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,200,?,1)', (7 << 24,))
        self.marker(110)
        self.marker(150, text=None, text_id=2)
        result = summarize_completion(self.db)
        self.assertEqual(result['ASTR_MPI_TESTALL_PENDING']['markers_during_kernel'], 1)
        self.assertEqual(result['ASTR_MPI_TESTALL_COMPLETE']['markers_during_kernel'], 1)

    def test_other_process_and_outside_kernel_are_not_overlap(self):
        self.db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,200,?,1)', (7 << 24,))
        self.marker(120, process=8)
        self.marker(99)
        self.marker(200)
        record = summarize_completion(self.db)['ASTR_MPI_TESTALL_PENDING']
        self.assertEqual(record['markers'], 3)
        self.assertEqual(record['markers_during_kernel'], 0)

    def test_concurrent_kernels_do_not_duplicate_marker_count(self):
        self.db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,200,?,1)',
                            [(7 << 24,), (7 << 24,)])
        self.marker(150)
        self.assertEqual(summarize_completion(self.db)['ASTR_MPI_TESTALL_PENDING']
                         ['markers_during_kernel'], 1)

    def test_complete_only_does_not_imply_pending_requests(self):
        self.marker(150, text='ASTR_MPI_TESTALL_COMPLETE')
        self.assertNotIn('ASTR_MPI_TESTALL_PENDING', summarize_completion(self.db))

    def test_absent_markers_are_not_invented(self):
        self.db.execute('DROP TABLE NVTX_EVENTS')
        self.assertEqual(summarize_completion(self.db), {})
