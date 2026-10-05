#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int armed, consumed;
static MPI_Fint private_comm = -1;

void astr_is4_test_arm(void) { armed = 1; }

static int phase(const char *name) {
    const char *value = getenv("ASTR_IS4_TEST_PHASE");
    return value && !strcmp(value, name);
}

static int root_rank(void) {
    const char *rank = getenv("OMPI_COMM_WORLD_RANK");
    return rank && !strcmp(rank,"0");
}

static int selected(const char *path, const char *name) {
    char expected[PATH_MAX], actual[PATH_MAX], cwd[PATH_MAX];
    const char *target = getenv("ASTR_IS4_TEST_TARGET");
    if (!target || snprintf(expected,sizeof(expected),"%s/%s",target,name)>=(int)sizeof(expected)) return 0;
    if (path[0]=='/') return !strcmp(path,expected);
    return getcwd(cwd,sizeof(cwd)) &&
        snprintf(actual,sizeof(actual),"%s/%s",cwd,path)<(int)sizeof(actual) && !strcmp(actual,expected);
}

static int reject_open(const char *path, int writing) {
    if (!writing || !root_rank() || consumed) return 0;
    if (phase("statistics_failure") && selected(path,"statistics.h5")) {
        consumed=1;
        dprintf(2,"ASTR_IS4_STATISTICS_ERROR_INJECTED\n");
        errno=EIO;
        return 1;
    }
    if ((phase("before_statistics") && selected(path,"statistics.h5")) ||
        (phase("before_render_control") && selected(path,"insitu_control.bin")) ||
        (phase("before_complete") && selected(path,"COMPLETE"))) {
        consumed=1;
        dprintf(2,"ASTR_IS4_INTERRUPTION phase=%s path=%s\n",getenv("ASTR_IS4_TEST_PHASE"),path);
        kill(getpid(),SIGKILL);
        _exit(137);
    }
    return 0;
}

#define INTERCEPT_OPEN(symbol) \
int symbol(const char *path,int flags,...) { \
    mode_t mode=0; \
    if ((flags&O_CREAT) || (flags&O_TMPFILE)==O_TMPFILE) { \
        va_list ap; va_start(ap,flags); mode=va_arg(ap,int); va_end(ap); \
    } \
    if (reject_open(path,(flags&O_ACCMODE)!=O_RDONLY)) return -1; \
    typedef int (*function)(const char*,int,...); \
    function real=(function)dlsym(RTLD_NEXT,#symbol); \
    if (!real) { errno=ENOSYS; return -1; } \
    return real(path,flags,mode); \
}
INTERCEPT_OPEN(open)
INTERCEPT_OPEN(open64)

#define INTERCEPT_FOPEN(symbol) \
FILE *symbol(const char *path,const char *mode) { \
    if (reject_open(path,mode[0]!='r' || strchr(mode,'+')!=NULL)) return NULL; \
    typedef FILE *(*function)(const char*,const char*); \
    function real=(function)dlsym(RTLD_NEXT,#symbol); \
    if (!real) { errno=ENOSYS; return NULL; } \
    return real(path,mode); \
}
INTERCEPT_FOPEN(fopen)
INTERCEPT_FOPEN(fopen64)

int MPI_Allreduce(const void *send,void *receive,int count,MPI_Datatype datatype,MPI_Op op,MPI_Comm comm) {
    int status=PMPI_Allreduce(send,receive,count,datatype,op,comm);
    if (armed && !consumed && phase("mpi_failure") && count==1 && datatype==MPI_INT && comm!=MPI_COMM_WORLD) {
        consumed=1;
        if (root_rank()) {
            dprintf(2,"ASTR_IS4_MPI_ERROR_INJECTED\n");
            return MPI_ERR_OTHER;
        }
    }
    return status;
}

void mpi_comm_dup_(MPI_Fint *comm,MPI_Fint *result,MPI_Fint *error) {
    typedef void (*function)(MPI_Fint*,MPI_Fint*,MPI_Fint*);
    function real=(function)dlsym(RTLD_NEXT,"mpi_comm_dup_");
    if (!real) _exit(99);
    real(comm,result,error);
    if (armed && !consumed && phase("nonfinite_sample") && *error==MPI_SUCCESS) private_comm=*result;
}

void mpi_sendrecv_(void *send,MPI_Fint *sc,MPI_Fint *st,MPI_Fint *destination,MPI_Fint *stag,
    void *receive,MPI_Fint *rc,MPI_Fint *rt,MPI_Fint *source,MPI_Fint *rtag,
    MPI_Fint *comm,MPI_Fint *status,MPI_Fint *error) {
    typedef void (*function)(void*,MPI_Fint*,MPI_Fint*,MPI_Fint*,MPI_Fint*,void*,MPI_Fint*,MPI_Fint*,
        MPI_Fint*,MPI_Fint*,MPI_Fint*,MPI_Fint*,MPI_Fint*);
    function real=(function)dlsym(RTLD_NEXT,"mpi_sendrecv_");
    if (!real) _exit(99);
    real(send,sc,st,destination,stag,receive,rc,rt,source,rtag,comm,status,error);
    if (armed && !consumed && phase("nonfinite_sample") && *comm==private_comm && *rc>0 &&
        MPI_Type_f2c(*rt)==MPI_DOUBLE_PRECISION && *error==MPI_SUCCESS) {
        consumed=1;
        if (root_rank()) {
            ((double*)receive)[0]=NAN;
            dprintf(2,"ASTR_IS4_NONFINITE_SAMPLE_INJECTED\n");
        }
    }
}
