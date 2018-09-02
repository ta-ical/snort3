#include <cstring>

#include <cuda_profiler_api.h>

#include "cuda_utils.h"

#include "pfac.h"
#include "pfac_match.cuh"
#include "pfac_table.h"

static inline void ConvertCaseEx (unsigned char *d, const uint8_t *s, unsigned m)
{
    unsigned i;
    for (i = 0; i < m; i++)
    {
        d[i] = pfac_xlatcase[s[i]];
    }
}


PFAC_status_t  PFAC_destroy( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    PFAC_freeResource( handle ) ;
    free( handle ) ;

    return PFAC_STATUS_SUCCESS ;
}

void  PFAC_freeResource( PFAC_handle_t handle )
{
    // resource of patterns
    if ( NULL != handle->rowPtr ){
        free( handle->rowPtr );
        handle->rowPtr = NULL ;
    }
    
    if ( NULL != handle->valPtr ){
        free( handle->valPtr );
        handle->valPtr = NULL ;
    }

    if ( NULL != handle->patternLen_table ){
        free( handle->patternLen_table ) ;
        handle->patternLen_table = NULL ;
    }
    
    if ( NULL != handle->patternID_table ){
        free( handle->patternID_table );
        handle->patternID_table = NULL ;
    }
    
    if ( NULL != handle->table_compact ){
        delete  handle->table_compact ;
        handle->table_compact = NULL ;
    }

    PFAC_freeTable( handle );
 
    handle->isPatternsReady = false ;
}

void  PFAC_freeTable( PFAC_handle_t handle )
{
    if ( NULL != handle->h_PFAC_table ){
        free( handle->h_PFAC_table ) ;
        handle->h_PFAC_table = NULL ;
    }

    // if ( NULL != handle->h_hashRowPtr ){
    //     free( handle->h_hashRowPtr );
    //     handle->h_hashRowPtr = NULL ;   
    // }
    
    // if ( NULL != handle->h_hashValPtr ){
    //     free( handle->h_hashValPtr );
    //     handle->h_hashValPtr = NULL ;   
    // }
    
    if ( NULL != handle->h_tableOfInitialState){
        free(handle->h_tableOfInitialState);
        handle->h_tableOfInitialState = NULL ; 
    }
    
    // free device resource
    if ( NULL != handle->d_PFAC_table ){
        cudaFree(handle->d_PFAC_table);
        handle->d_PFAC_table= NULL ;
    }
    
    // if ( NULL != handle->d_hashRowPtr ){
    //     cudaFree( handle->d_hashRowPtr );
    //     handle->d_hashRowPtr = NULL ;
    // }

    // if ( NULL != handle->d_hashValPtr ){
    //     cudaFree( handle->d_hashValPtr );
    //     handle->d_hashValPtr = NULL ;   
    // }
    
    if ( NULL != handle->d_tableOfInitialState ){
        cudaFree(handle->d_tableOfInitialState);
        handle->d_tableOfInitialState = NULL ;
    }

    if ( NULL != handle->d_input_string ){
        cudaFree(handle->d_input_string);
        handle->d_input_string = NULL ;
    }

    if ( NULL != handle->d_matched_result ){
        cudaFree(handle->d_matched_result);
        handle->d_matched_result = NULL ;
    }
}

PFAC_status_t PFAC_tex_mutex_lock(PFAC_handle_t handle)
{
    try
    {
        handle->__pfac_tex_mutex.lock();
    }
    catch (const system_error &e)
    {
        return PFAC_STATUS_MUTEX_ERROR;
    }

    return PFAC_STATUS_SUCCESS;
}

PFAC_status_t PFAC_tex_mutex_unlock(PFAC_handle_t handle)
{
    try
    {
        handle->__pfac_tex_mutex.unlock();
    }
    catch (const system_error &e)
    {
        return PFAC_STATUS_MUTEX_ERROR;
    }

    return PFAC_STATUS_SUCCESS;
}

PFAC_status_t  PFAC_create( PFAC_handle_t handle )
{
    int device ;
    cudaError_t cuda_status = cudaGetDevice( &device ) ;
    if ( cudaSuccess != cuda_status ){
        return (PFAC_status_t)cuda_status ;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    PFAC_PRINTF("major = %d, minor = %d, name=%s\n", deviceProp.major, deviceProp.minor, deviceProp.name );

    int device_no = 10*deviceProp.major + deviceProp.minor ;
    handle->device_no = device_no ;

    // Find entry point of PFAC_kernel
    handle->kernel_ptr = (PFAC_kernel_protoType) PFAC_kernel_timeDriven_wrapper;
    if ( NULL == handle->kernel_ptr ){
        PFAC_PRINTF("Error: cannot load PFAC_kernel_timeDriven_wrapper, error = %s\n", "" );
        return PFAC_STATUS_INTERNAL_ERROR ;
    }   

    // allocate memory for input string and result
    // basic unit of d_input_string is integer
    size_t ava, total;
    cudaMemGetInfo (&ava, &total); 	
    PFAC_PRINTF("Available mem %u from %u\n", ava, total);
    cudaError_t cuda_status1 = cudaMalloc((void **) &(handle->d_input_string),     MAX_BUFFER_SIZE*sizeof(char) );
    cudaError_t cuda_status2 = cudaMalloc((void **) &(handle->d_matched_result),    MAX_BUFFER_SIZE*sizeof(int) );
    cudaError_t cuda_status3 = cudaMallocHost((void**) &(handle->h_input_string),  MAX_BUFFER_SIZE*sizeof(char) );
    cudaError_t cuda_status4 = cudaMallocHost((void**) &(handle->h_matched_result), MAX_BUFFER_SIZE*sizeof(int) );
    int *h_matched_result = (int *) malloc ( MAX_BUFFER_SIZE * sizeof(int) );
    int *h_num_matched = (int *) malloc ( THREAD_BLOCK_SIZE * sizeof(int) );
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) || 
         (cudaSuccess != cuda_status3) || (cudaSuccess != cuda_status4)) {
        if ( NULL != handle->d_input_string   ) { cudaFree(handle->d_input_string); }
        if ( NULL != handle->d_matched_result ) { cudaFree(handle->d_matched_result); }
        if ( NULL != handle->h_input_string   ) { cudaFree(handle->h_input_string); }
        if ( NULL != handle->h_matched_result ) { cudaFree(handle->h_matched_result); }
        return PFAC_STATUS_CUDA_ALLOC_FAILED;
    }

    return PFAC_STATUS_SUCCESS ;
}



PFAC_STRUCT * pfacNew (const MpseAgent* agent)
{
    PFAC_handle_t handle = (PFAC_handle_t) calloc( 1, sizeof(PFAC_STRUCT) ) ;
    if ( handle == NULL ){
        PFAC_PRINTF("Error: cannot initialize handler, error = %s\n", PFAC_getErrorString(PFAC_STATUS_ALLOC_FAILED));
        return NULL;
    }

    memset( handle, 0, sizeof(PFAC_STRUCT) ) ;

    PFAC_status_t status = PFAC_create( handle );
    if ( status != PFAC_STATUS_SUCCESS )
    {
        PFAC_PRINTF("Error: cannot initialize handler, error = %s\n", PFAC_getErrorString(status));
        return NULL;
    }
    
    if (agent != NULL) 
    {
        handle->userfree              = agent->user_free;
        handle->optiontreefree        = agent->tree_free;
        handle->neg_list_free         = agent->list_free;
    }

    return (PFAC_STRUCT *) handle;
}

void pfacFree ( PFAC_STRUCT * pfac )
{
    PFAC_handle_t handle = (PFAC_handle_t) pfac;
    PFAC_status_t status;
    
    bool texture_on = (PFAC_TEXTURE_ON == handle->textureMode );
    if ( texture_on && handle->sizeOfTableInBytes > 0 ){
        status = PFAC_unbindTexture(handle);
        if ( status != PFAC_STATUS_SUCCESS )
        {
            PFAC_PRINTF("Error: cannot unbind texture, error = %s\n", PFAC_getErrorString(status));
        }
    }

    status = PFAC_destroy( handle ) ;
    if ( status != PFAC_STATUS_SUCCESS )
    {
        PFAC_PRINTF("Error: cannot deinitialize handler, error = %s\n", PFAC_getErrorString(status));
    }

    PFAC_PRINTF("Deallocation succeed\n");
}

int pfacAddPattern ( 
    PFAC_STRUCT * p, const uint8_t* pat, unsigned n, bool nocase,
    bool negative, void* user )
{
    // PFAC_PRINTF("Pat length: %u\n", n);

    PFAC_PATTERN * plist;
    plist = (PFAC_PATTERN *) calloc (1, sizeof (PFAC_PATTERN));
    plist->patrn = (uint8_t *) calloc (n, 1);
    ConvertCaseEx (plist->patrn, pat, n);
    plist->casepatrn = (uint8_t *) calloc (n, 1);
    memcpy (plist->casepatrn, pat, n);

    plist->udata = (PFAC_USERDATA*)user;
    plist->n = n;
    plist->nocase = nocase;
    plist->negative = negative;
    // plist->offset = offset;
    // plist->depth = depth;
    // plist->iid = iid;
    plist->next = p->pfacPatterns;
    
    p->pfacPatterns = plist;
    p->numOfPatterns++;
    p->max_numOfStates += n + 1;
    return 0;
}


int pfacCompile ( SnortConfig * config, PFAC_STRUCT * pfac )
{
    int max_numOfStates = ++pfac->max_numOfStates;

    // Allocate a buffer to contains all patterns
    pfac->valPtr = (char*)calloc(max_numOfStates, sizeof( char ));
    if (NULL == pfac->valPtr) {
        return PFAC_STATUS_ALLOC_FAILED;
    }

    PFAC_status_t status = PFAC_fillPatternTable((PFAC_handle_t) pfac);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_fillPatternTable, %s\n", PFAC_getErrorString(status) );
        PFAC_freeResource( (PFAC_handle_t) pfac );
        return 0;
    }

    status = PFAC_prepareTable((PFAC_handle_t) pfac);
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_prepareTable, %s\n", PFAC_getErrorString(status) );
        PFAC_freeResource( (PFAC_handle_t) pfac );
        return 0;
    }

    return 0;
}

int pfacSearch ( 
    PFAC_STRUCT * pfac, const uint8_t* T, int n, MpseMatch match,
    void* context, int* current_state )
{
    PFAC_PRINTF("Stream length: %u\n", n);

    int nfound = 0;
    PFAC_handle_t handle = (PFAC_handle_t) pfac;

    memcpy(handle->h_input_string, T, n*sizeof(char));

    PFAC_status_t status = PFAC_matchFromHost( handle, n ) ;
    if ( status != PFAC_STATUS_SUCCESS ) {
        PFAC_PRINTF("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(status) );
        return 0;
    }

    #pragma omp parallel for reduction (+:nfound)
    for (int i = 0; i < THREAD_BLOCK_SIZE; ++i)
    {
        nfound += handle->h_matched_result[i];
    }

    return nfound;
}

int pfacPatternCount ( PFAC_STRUCT * pfac )
{
    return pfac->numOfPatterns;
}

int pfacPrintDetailInfo(PFAC_STRUCT * p)
{
    if(p)
        p = p;
    return 0;
}

int pfacPrintSummaryInfo(void)
{
    // SPFAC_STRUCT2 * p = &summary.spfac;

    // if( !summary.num_states )
    //     return;

    // PFAC_PRINTF("+--[Pattern Matcher:Aho-Corasick Summary]----------------------\n");
    // PFAC_PRINTF("| Alphabet Size    : %d Chars\n",p->spfacAlphabetSize);
    // PFAC_PRINTF("| Sizeof State     : %d bytes\n",sizeof(acstate_t));
    // PFAC_PRINTF("| Storage Format   : %s \n",sf[ p->spfacFormat ]);
    // PFAC_PRINTF("| Num States       : %d\n",summary.num_states);
    // PFAC_PRINTF("| Num Transitions  : %d\n",summary.num_transitions);
    // PFAC_PRINTF("| State Density    : %.1f%%\n",100.0*(double)summary.num_transitions/(summary.num_states*p->spfacAlphabetSize));
    // PFAC_PRINTF("| Finite Automatum : %s\n", fsa[p->spfacFSA]);
    // if( max_memory < 1024*1024 )
    //     PFAC_PRINTF("| Memory           : %.2fKbytes\n", (float)max_memory/1024 );
    // else
    //     PFAC_PRINTF("| Memory           : %.2fMbytes\n", (float)max_memory/(1024*1024) );
    // PFAC_PRINTF("+-------------------------------------------------------------\n");

    return 0;
}
