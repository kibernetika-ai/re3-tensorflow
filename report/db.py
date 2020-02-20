import logging
import sqlite3


LOG = logging.getLogger(__name__)


initial_create = """
-- classes with meta
create table IF NOT EXISTS classes 
(
    id varchar(50) constraint classes_pk primary key,
    name TEXT not null,
    age integer,
    gender varchar(20),
    meta TEXT
);

-- classes linking to faces in hdf5
create table IF NOT EXISTS classes_faces 
(
  video    TEXT,
  face_idx integer,
  frame_num integer,
  class_id varchar(50) not null,
  constraint classes_faces_pk primary key (video, class_id, face_idx)
);

create index IF NOT EXISTS classes_faces_class_id_idx on classes_faces (class_id);

-- classes appearance intervals
create table IF NOT EXISTS intervals
(
    class_id varchar(50) not null,
    start REAL not null,
    end REAL not null
);
create index IF NOT EXISTS intervals_class_id_idx on intervals (class_id);

-- classes attention intervals
create table IF NOT EXISTS attention
(
    class_id varchar(50) not null,
    start REAL not null,
    end REAL not null
);
create index IF NOT EXISTS attention_class_id_idx on attention (class_id);

-- classes emotions index
create table IF NOT EXISTS emotions
(
    class_id varchar(50) not null,
    time REAL not null,
    emotion varchar(20)
);
create index IF NOT EXISTS emotions_class_id_idx on emotions (class_id);
create index IF NOT EXISTS emotions_time_idx on emotions (time);
"""

clear_report_tables = """
drop table IF EXISTS emotions;
drop table IF EXISTS intervals;
drop table IF EXISTS attention;
drop table IF EXISTS classes;
drop table IF EXISTS classes_faces;
"""


class ReportDB(object):
    def __init__(self, db_file, clear_report=False):
        self.db_file = db_file
        self.conn = self.create_conn(db_file)
        self.cursor = self.conn.cursor()
        self.create_tables(clear_report=clear_report)
        self.exec('PRAGMA journal_mode=WAL')

    @staticmethod
    def create_conn(db_file: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_file)
        return conn

    def begin_transaction(self):
        self.exec('BEGIN TRANSACTION')

    def end_transaction(self):
        self.exec('COMMIT')

    def insert_data(self, table, data_dict, commit=True):
        keys = []
        values = []
        for k, v in data_dict.items():
            keys.append(str(k))
            values.append(v)
        sql = f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' for v in values)})"

        self.exec(sql, values, commit=commit)

    def create_tables(self, clear_report=False):
        cursor = self.cursor
        if clear_report:
            LOG.info("[DB] Clear report tables")
            cursor.executescript(clear_report_tables)
        cursor.executescript(initial_create)

        self.conn.commit()

    def exec(self, sql: str, params=None, commit=True):
        LOG.debug(f"[DB] Executing {sql}; params={params}")
        self.cursor.execute(sql, params if params else [])
        if commit:
            self.conn.commit()
