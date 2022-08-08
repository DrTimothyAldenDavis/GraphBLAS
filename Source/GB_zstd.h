//------------------------------------------------------------------------------
// GB_zstd.h: definitions for a wrapper for the ZSTD compression library
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// It's possible that the user application has its own copy of the ZSTD library,
// which wouldn't be using the SuiteSparse:GraphBLAS memory allocator.  To
// avoid any conflict between multiple copies of the ZSTD library, all global
// symbols ZSTD_* are renamed to GBZSTD (ZSTD_*), via #defines below.

#ifndef GB_ZSTD_H 
#define GB_ZSTD_H 

// ZSTD has its own GB macro, so #undefine the GraphBLAS one, and use GBZSTD
// to rename the ZSTD functions.
#undef GB

#ifdef GBRENAME
    #define GBZSTD(x) GB_EVAL2 (GM_, x)
#else
    #define GBZSTD(x) GB_EVAL2 (GB_, x)
#endif

//------------------------------------------------------------------------------
// methods called directly by GraphBLAS
//------------------------------------------------------------------------------

// size_t ZSTD_compressBound (size_t s) : returns the maximum size for the
// compression of a block of s bytes.
#define ZSTD_compressBound  GBZSTD (ZSTD_compressBound)

// size_t ZSTD_compress (void *dst, size_t dstCap, const void *src, size_t
// srcSize, int level) : compresses the uncompressed src block of size srcSize
// into the output buffer dst of size dstCap.  Returns the compressed size
// written into dst (<= dstCap), or an error code if it fails.
#define ZSTD_compress       GBZSTD (ZSTD_compress)

// size_t ZSTD_decompress (void *dst, size_t dstCap, const void *src, size_t
// compressedSize) : decompresses the compressed src block of size
// compressedSize into the dst block of size dstCap.  Returns the # of bytes
// written to dst (<= dstCap), or an error code if it fails.
#define ZSTD_decompress     GBZSTD (ZSTD_decompress)

//------------------------------------------------------------------------------
// ensure that ZSTD_malloc, ZSTD_calloc, and ZSTD_free are used.
//------------------------------------------------------------------------------

// ZSTD will use these 3 functions in place of malloc, calloc, and free.  They
// are defined in GB_zstd.c, and rely on the malloc and free methods provided by
// the user application to GraphBLAS by GrB_init or GxB_init.

#define ZSTD_DEPS_MALLOC
#define ZSTD_malloc  GBZSTD (ZSTD_malloc)
#define ZSTD_calloc  GBZSTD (ZSTD_calloc)
#define ZSTD_free    GBZSTD (ZSTD_free)
void *ZSTD_malloc (size_t s) ;
void *ZSTD_calloc (size_t n, size_t s) ;
void  ZSTD_free (void *p) ;

//------------------------------------------------------------------------------
// methods not directly used, or not used at all by GraphBLAS
//------------------------------------------------------------------------------

#define ZSTD_adjustCParams                       \
GBZSTD (ZSTD_adjustCParams)
#define ZSTD_buildBlockEntropyStats              \
GBZSTD (ZSTD_buildBlockEntropyStats)
#define ZSTD_buildCTable                         \
GBZSTD (ZSTD_buildCTable)
#define ZSTD_buildFSETable                       \
GBZSTD (ZSTD_buildFSETable)
#define ZSTD_CCtx_getParameter                   \
GBZSTD (ZSTD_CCtx_getParameter)
#define ZSTD_CCtx_loadDictionary                 \
GBZSTD (ZSTD_CCtx_loadDictionary)
#define ZSTD_CCtx_loadDictionary_advanced        \
GBZSTD (ZSTD_CCtx_loadDictionary_advanced)
#define ZSTD_CCtx_loadDictionary_byReference     \
GBZSTD (ZSTD_CCtx_loadDictionary_byReference)
#define ZSTD_CCtxParams_getParameter             \
GBZSTD (ZSTD_CCtxParams_getParameter)
#define ZSTD_CCtxParams_init                     \
GBZSTD (ZSTD_CCtxParams_init)
#define ZSTD_CCtxParams_init_advanced            \
GBZSTD (ZSTD_CCtxParams_init_advanced)
#define ZSTD_CCtxParams_reset                    \
GBZSTD (ZSTD_CCtxParams_reset)
#define ZSTD_CCtxParams_setParameter             \
GBZSTD (ZSTD_CCtxParams_setParameter)
#define ZSTD_CCtx_refCDict                       \
GBZSTD (ZSTD_CCtx_refCDict)
#define ZSTD_CCtx_refPrefix                      \
GBZSTD (ZSTD_CCtx_refPrefix)
#define ZSTD_CCtx_refPrefix_advanced             \
GBZSTD (ZSTD_CCtx_refPrefix_advanced)
#define ZSTD_CCtx_refThreadPool                  \
GBZSTD (ZSTD_CCtx_refThreadPool)
#define ZSTD_CCtx_reset                          \
GBZSTD (ZSTD_CCtx_reset)
#define ZSTD_CCtx_setParameter                   \
GBZSTD (ZSTD_CCtx_setParameter)
#define ZSTD_CCtx_setParametersUsingCCtxParams   \
GBZSTD (ZSTD_CCtx_setParametersUsingCCtxParams)
#define ZSTD_CCtx_setPledgedSrcSize              \
GBZSTD (ZSTD_CCtx_setPledgedSrcSize)
#define ZSTD_CCtx_trace                          \
GBZSTD (ZSTD_CCtx_trace)
#define ZSTD_checkContinuity                     \
GBZSTD (ZSTD_checkContinuity)
#define ZSTD_checkCParams                        \
GBZSTD (ZSTD_checkCParams)
#define ZSTD_compress2                           \
GBZSTD (ZSTD_compress2)
#define ZSTD_compress_advanced                   \
GBZSTD (ZSTD_compress_advanced)
#define ZSTD_compress_advanced_internal          \
GBZSTD (ZSTD_compress_advanced_internal)
#define ZSTD_compressBegin                       \
GBZSTD (ZSTD_compressBegin)
#define ZSTD_compressBegin_advanced              \
GBZSTD (ZSTD_compressBegin_advanced)
#define ZSTD_compressBegin_advanced_internal     \
GBZSTD (ZSTD_compressBegin_advanced_internal)
#define ZSTD_compressBegin_usingCDict            \
GBZSTD (ZSTD_compressBegin_usingCDict)
#define ZSTD_compressBegin_usingCDict_advanced   \
GBZSTD (ZSTD_compressBegin_usingCDict_advanced)
#define ZSTD_compressBegin_usingDict             \
GBZSTD (ZSTD_compressBegin_usingDict)
#define ZSTD_compressBlock                       \
GBZSTD (ZSTD_compressBlock)
#define ZSTD_compressBlock_btlazy2               \
GBZSTD (ZSTD_compressBlock_btlazy2)
#define ZSTD_compressBlock_btlazy2_dictMatchState \
GBZSTD (ZSTD_compressBlock_btlazy2_dictMatchState)
#define ZSTD_compressBlock_btlazy2_extDict       \
GBZSTD (ZSTD_compressBlock_btlazy2_extDict)
#define ZSTD_compressBlock_btopt                 \
GBZSTD (ZSTD_compressBlock_btopt)
#define ZSTD_compressBlock_btopt_dictMatchState  \
GBZSTD (ZSTD_compressBlock_btopt_dictMatchState)
#define ZSTD_compressBlock_btopt_extDict         \
GBZSTD (ZSTD_compressBlock_btopt_extDict)
#define ZSTD_compressBlock_btultra               \
GBZSTD (ZSTD_compressBlock_btultra)
#define ZSTD_compressBlock_btultra2              \
GBZSTD (ZSTD_compressBlock_btultra2)
#define ZSTD_compressBlock_btultra_dictMatchState \
GBZSTD (ZSTD_compressBlock_btultra_dictMatchState)
#define ZSTD_compressBlock_btultra_extDict       \
GBZSTD (ZSTD_compressBlock_btultra_extDict)
#define ZSTD_compressBlock_doubleFast            \
GBZSTD (ZSTD_compressBlock_doubleFast)
#define ZSTD_compressBlock_doubleFast_dictMatchState \
GBZSTD (ZSTD_compressBlock_doubleFast_dictMatchState)
#define ZSTD_compressBlock_doubleFast_extDict    \
GBZSTD (ZSTD_compressBlock_doubleFast_extDict)
#define ZSTD_compressBlock_fast                  \
GBZSTD (ZSTD_compressBlock_fast)
#define ZSTD_compressBlock_fast_dictMatchState   \
GBZSTD (ZSTD_compressBlock_fast_dictMatchState)
#define ZSTD_compressBlock_fast_extDict          \
GBZSTD (ZSTD_compressBlock_fast_extDict)
#define ZSTD_compressBlock_greedy                \
GBZSTD (ZSTD_compressBlock_greedy)
#define ZSTD_compressBlock_greedy_dedicatedDictSearch \
GBZSTD (ZSTD_compressBlock_greedy_dedicatedDictSearch)
#define ZSTD_compressBlock_greedy_dedicatedDictSearch_row \
GBZSTD (ZSTD_compressBlock_greedy_dedicatedDictSearch_row)
#define ZSTD_compressBlock_greedy_dictMatchState \
GBZSTD (ZSTD_compressBlock_greedy_dictMatchState)
#define ZSTD_compressBlock_greedy_dictMatchState_row \
GBZSTD (ZSTD_compressBlock_greedy_dictMatchState_row)
#define ZSTD_compressBlock_greedy_extDict        \
GBZSTD (ZSTD_compressBlock_greedy_extDict)
#define ZSTD_compressBlock_greedy_extDict_row    \
GBZSTD (ZSTD_compressBlock_greedy_extDict_row)
#define ZSTD_compressBlock_greedy_row            \
GBZSTD (ZSTD_compressBlock_greedy_row)
#define ZSTD_compressBlock_lazy                  \
GBZSTD (ZSTD_compressBlock_lazy)
#define ZSTD_compressBlock_lazy2                 \
GBZSTD (ZSTD_compressBlock_lazy2)
#define ZSTD_compressBlock_lazy2_dedicatedDictSearch \
GBZSTD (ZSTD_compressBlock_lazy2_dedicatedDictSearch)
#define ZSTD_compressBlock_lazy2_dedicatedDictSearch_row \
GBZSTD (ZSTD_compressBlock_lazy2_dedicatedDictSearch_row)
#define ZSTD_compressBlock_lazy2_dictMatchState  \
GBZSTD (ZSTD_compressBlock_lazy2_dictMatchState)
#define ZSTD_compressBlock_lazy2_dictMatchState_row \
GBZSTD (ZSTD_compressBlock_lazy2_dictMatchState_row)
#define ZSTD_compressBlock_lazy2_extDict         \
GBZSTD (ZSTD_compressBlock_lazy2_extDict)
#define ZSTD_compressBlock_lazy2_extDict_row     \
GBZSTD (ZSTD_compressBlock_lazy2_extDict_row)
#define ZSTD_compressBlock_lazy2_row             \
GBZSTD (ZSTD_compressBlock_lazy2_row)
#define ZSTD_compressBlock_lazy_dedicatedDictSearch \
GBZSTD (ZSTD_compressBlock_lazy_dedicatedDictSearch)
#define ZSTD_compressBlock_lazy_dedicatedDictSearch_row \
GBZSTD (ZSTD_compressBlock_lazy_dedicatedDictSearch_row)
#define ZSTD_compressBlock_lazy_dictMatchState   \
GBZSTD (ZSTD_compressBlock_lazy_dictMatchState)
#define ZSTD_compressBlock_lazy_dictMatchState_row \
GBZSTD (ZSTD_compressBlock_lazy_dictMatchState_row)
#define ZSTD_compressBlock_lazy_extDict          \
GBZSTD (ZSTD_compressBlock_lazy_extDict)
#define ZSTD_compressBlock_lazy_extDict_row      \
GBZSTD (ZSTD_compressBlock_lazy_extDict_row)
#define ZSTD_compressBlock_lazy_row              \
GBZSTD (ZSTD_compressBlock_lazy_row)
#define ZSTD_compressCCtx                        \
GBZSTD (ZSTD_compressCCtx)
#define ZSTD_compressContinue                    \
GBZSTD (ZSTD_compressContinue)
#define ZSTD_compressEnd                         \
GBZSTD (ZSTD_compressEnd)
#define ZSTD_compressLiterals                    \
GBZSTD (ZSTD_compressLiterals)
#define ZSTD_compressRleLiteralsBlock            \
GBZSTD (ZSTD_compressRleLiteralsBlock)
#define ZSTD_compressSequences                   \
GBZSTD (ZSTD_compressSequences)
#define ZSTD_compressStream                      \
GBZSTD (ZSTD_compressStream)
#define ZSTD_compressStream2                     \
GBZSTD (ZSTD_compressStream2)
#define ZSTD_compressStream2_simpleArgs          \
GBZSTD (ZSTD_compressStream2_simpleArgs)
#define ZSTD_compressSuperBlock                  \
GBZSTD (ZSTD_compressSuperBlock)
#define ZSTD_compress_usingCDict                 \
GBZSTD (ZSTD_compress_usingCDict)
#define ZSTD_compress_usingCDict_advanced        \
GBZSTD (ZSTD_compress_usingCDict_advanced)
#define ZSTD_compress_usingDict                  \
GBZSTD (ZSTD_compress_usingDict)
#define ZSTD_copyCCtx                            \
GBZSTD (ZSTD_copyCCtx)
#define ZSTD_copyDCtx                            \
GBZSTD (ZSTD_copyDCtx)
#define ZSTD_copyDDictParameters                 \
GBZSTD (ZSTD_copyDDictParameters)
#define ZSTD_cParam_getBounds                    \
GBZSTD (ZSTD_cParam_getBounds)
#define ZSTD_createCCtx                          \
GBZSTD (ZSTD_createCCtx)
#define ZSTD_createCCtx_advanced                 \
GBZSTD (ZSTD_createCCtx_advanced)
#define ZSTD_createCCtxParams                    \
GBZSTD (ZSTD_createCCtxParams)
#define ZSTD_createCDict                         \
GBZSTD (ZSTD_createCDict)
#define ZSTD_createCDict_advanced                \
GBZSTD (ZSTD_createCDict_advanced)
#define ZSTD_createCDict_advanced2               \
GBZSTD (ZSTD_createCDict_advanced2)
#define ZSTD_createCDict_byReference             \
GBZSTD (ZSTD_createCDict_byReference)
#define ZSTD_createCStream                       \
GBZSTD (ZSTD_createCStream)
#define ZSTD_createCStream_advanced              \
GBZSTD (ZSTD_createCStream_advanced)
#define ZSTD_createDCtx                          \
GBZSTD (ZSTD_createDCtx)
#define ZSTD_createDCtx_advanced                 \
GBZSTD (ZSTD_createDCtx_advanced)
#define ZSTD_createDDict                         \
GBZSTD (ZSTD_createDDict)
#define ZSTD_createDDict_advanced                \
GBZSTD (ZSTD_createDDict_advanced)
#define ZSTD_createDDict_byReference             \
GBZSTD (ZSTD_createDDict_byReference)
#define ZSTD_createDStream                       \
GBZSTD (ZSTD_createDStream)
#define ZSTD_createDStream_advanced              \
GBZSTD (ZSTD_createDStream_advanced)
#define ZSTD_crossEntropyCost                    \
GBZSTD (ZSTD_crossEntropyCost)
#define ZSTD_CStreamInSize                       \
GBZSTD (ZSTD_CStreamInSize)
#define ZSTD_CStreamOutSize                      \
GBZSTD (ZSTD_CStreamOutSize)
#define ZSTD_customCalloc                        \
GBZSTD (ZSTD_customCalloc)
#define ZSTD_customFree                          \
GBZSTD (ZSTD_customFree)
#define ZSTD_customMalloc                        \
GBZSTD (ZSTD_customMalloc)
#define ZSTD_cycleLog                            \
GBZSTD (ZSTD_cycleLog)
#define ZSTD_DCtx_getParameter                   \
GBZSTD (ZSTD_DCtx_getParameter)
#define ZSTD_DCtx_loadDictionary                 \
GBZSTD (ZSTD_DCtx_loadDictionary)
#define ZSTD_DCtx_loadDictionary_advanced        \
GBZSTD (ZSTD_DCtx_loadDictionary_advanced)
#define ZSTD_DCtx_loadDictionary_byReference     \
GBZSTD (ZSTD_DCtx_loadDictionary_byReference)
#define ZSTD_DCtx_refDDict                       \
GBZSTD (ZSTD_DCtx_refDDict)
#define ZSTD_DCtx_refPrefix                      \
GBZSTD (ZSTD_DCtx_refPrefix)
#define ZSTD_DCtx_refPrefix_advanced             \
GBZSTD (ZSTD_DCtx_refPrefix_advanced)
#define ZSTD_DCtx_reset                          \
GBZSTD (ZSTD_DCtx_reset)
#define ZSTD_DCtx_setFormat                      \
GBZSTD (ZSTD_DCtx_setFormat)
#define ZSTD_DCtx_setMaxWindowSize               \
GBZSTD (ZSTD_DCtx_setMaxWindowSize)
#define ZSTD_DCtx_setParameter                   \
GBZSTD (ZSTD_DCtx_setParameter)
#define ZSTD_DDict_dictContent                   \
GBZSTD (ZSTD_DDict_dictContent)
#define ZSTD_DDict_dictSize                      \
GBZSTD (ZSTD_DDict_dictSize)
#define ZSTD_decodeLiteralsBlock                 \
GBZSTD (ZSTD_decodeLiteralsBlock)
#define ZSTD_decodeSeqHeaders                    \
GBZSTD (ZSTD_decodeSeqHeaders)
#define ZSTD_decodingBufferSize_min              \
GBZSTD (ZSTD_decodingBufferSize_min)
#define ZSTD_decompressBegin                     \
GBZSTD (ZSTD_decompressBegin)
#define ZSTD_decompressBegin_usingDDict          \
GBZSTD (ZSTD_decompressBegin_usingDDict)
#define ZSTD_decompressBegin_usingDict           \
GBZSTD (ZSTD_decompressBegin_usingDict)
#define ZSTD_decompressBlock                     \
GBZSTD (ZSTD_decompressBlock)
#define ZSTD_decompressBlock_internal            \
GBZSTD (ZSTD_decompressBlock_internal)
#define ZSTD_decompressBound                     \
GBZSTD (ZSTD_decompressBound)
#define ZSTD_decompressContinue                  \
GBZSTD (ZSTD_decompressContinue)
#define ZSTD_decompressDCtx                      \
GBZSTD (ZSTD_decompressDCtx)
#define ZSTD_decompressStream                    \
GBZSTD (ZSTD_decompressStream)
#define ZSTD_decompressStream_simpleArgs         \
GBZSTD (ZSTD_decompressStream_simpleArgs)
#define ZSTD_decompress_usingDDict               \
GBZSTD (ZSTD_decompress_usingDDict)
#define ZSTD_decompress_usingDict                \
GBZSTD (ZSTD_decompress_usingDict)
#define ZSTD_dedicatedDictSearch_lazy_loadDictionary \
GBZSTD (ZSTD_dedicatedDictSearch_lazy_loadDictionary)
#define ZSTD_defaultCLevel                       \
GBZSTD (ZSTD_defaultCLevel)
#define ZSTD_dParam_getBounds                    \
GBZSTD (ZSTD_dParam_getBounds)
#define ZSTD_DStreamInSize                       \
GBZSTD (ZSTD_DStreamInSize)
#define ZSTD_DStreamOutSize                      \
GBZSTD (ZSTD_DStreamOutSize)
#define ZSTD_encodeSequences                     \
GBZSTD (ZSTD_encodeSequences)
#define ZSTD_endStream                           \
GBZSTD (ZSTD_endStream)
#define ZSTD_estimateCCtxSize                    \
GBZSTD (ZSTD_estimateCCtxSize)
#define ZSTD_estimateCCtxSize_usingCCtxParams    \
GBZSTD (ZSTD_estimateCCtxSize_usingCCtxParams)
#define ZSTD_estimateCCtxSize_usingCParams       \
GBZSTD (ZSTD_estimateCCtxSize_usingCParams)
#define ZSTD_estimateCDictSize                   \
GBZSTD (ZSTD_estimateCDictSize)
#define ZSTD_estimateCDictSize_advanced          \
GBZSTD (ZSTD_estimateCDictSize_advanced)
#define ZSTD_estimateCStreamSize                 \
GBZSTD (ZSTD_estimateCStreamSize)
#define ZSTD_estimateCStreamSize_usingCCtxParams \
GBZSTD (ZSTD_estimateCStreamSize_usingCCtxParams)
#define ZSTD_estimateCStreamSize_usingCParams    \
GBZSTD (ZSTD_estimateCStreamSize_usingCParams)
#define ZSTD_estimateDCtxSize                    \
GBZSTD (ZSTD_estimateDCtxSize)
#define ZSTD_estimateDDictSize                   \
GBZSTD (ZSTD_estimateDDictSize)
#define ZSTD_estimateDStreamSize                 \
GBZSTD (ZSTD_estimateDStreamSize)
#define ZSTD_estimateDStreamSize_fromFrame       \
GBZSTD (ZSTD_estimateDStreamSize_fromFrame)
#define ZSTD_fillDoubleHashTable                 \
GBZSTD (ZSTD_fillDoubleHashTable)
#define ZSTD_fillHashTable                       \
GBZSTD (ZSTD_fillHashTable)
#define ZSTD_findDecompressedSize                \
GBZSTD (ZSTD_findDecompressedSize)
#define ZSTD_findFrameCompressedSize             \
GBZSTD (ZSTD_findFrameCompressedSize)
#define ZSTD_flushStream                         \
GBZSTD (ZSTD_flushStream)
#define ZSTD_frameHeaderSize                     \
GBZSTD (ZSTD_frameHeaderSize)
#define ZSTD_freeCCtx                            \
GBZSTD (ZSTD_freeCCtx)
#define ZSTD_freeCCtxParams                      \
GBZSTD (ZSTD_freeCCtxParams)
#define ZSTD_freeCDict                           \
GBZSTD (ZSTD_freeCDict)
#define ZSTD_freeCStream                         \
GBZSTD (ZSTD_freeCStream)
#define ZSTD_freeDCtx                            \
GBZSTD (ZSTD_freeDCtx)
#define ZSTD_freeDDict                           \
GBZSTD (ZSTD_freeDDict)
#define ZSTD_freeDStream                         \
GBZSTD (ZSTD_freeDStream)
#define ZSTD_fseBitCost                          \
GBZSTD (ZSTD_fseBitCost)
#define ZSTD_generateSequences                   \
GBZSTD (ZSTD_generateSequences)
#define ZSTD_getBlockSize                        \
GBZSTD (ZSTD_getBlockSize)
#define ZSTD_getcBlockSize                       \
GBZSTD (ZSTD_getcBlockSize)
#define ZSTD_getCParams                          \
GBZSTD (ZSTD_getCParams)
#define ZSTD_getCParamsFromCCtxParams            \
GBZSTD (ZSTD_getCParamsFromCCtxParams)
#define ZSTD_getCParamsFromCDict                 \
GBZSTD (ZSTD_getCParamsFromCDict)
#define ZSTD_getDecompressedSize                 \
GBZSTD (ZSTD_getDecompressedSize)
#define ZSTD_getDictID_fromCDict                 \
GBZSTD (ZSTD_getDictID_fromCDict)
#define ZSTD_getDictID_fromDDict                 \
GBZSTD (ZSTD_getDictID_fromDDict)
#define ZSTD_getDictID_fromDict                  \
GBZSTD (ZSTD_getDictID_fromDict)
#define ZSTD_getDictID_fromFrame                 \
GBZSTD (ZSTD_getDictID_fromFrame)
#define ZSTD_getErrorCode                        \
GBZSTD (ZSTD_getErrorCode)
#define ZSTD_getErrorName                        \
GBZSTD (ZSTD_getErrorName)
#define ZSTD_getErrorString                      \
GBZSTD (ZSTD_getErrorString)
#define ZSTD_getFrameContentSize                 \
GBZSTD (ZSTD_getFrameContentSize)
#define ZSTD_getFrameHeader                      \
GBZSTD (ZSTD_getFrameHeader)
#define ZSTD_getFrameHeader_advanced             \
GBZSTD (ZSTD_getFrameHeader_advanced)
#define ZSTD_getFrameProgression                 \
GBZSTD (ZSTD_getFrameProgression)
#define ZSTD_getParams                           \
GBZSTD (ZSTD_getParams)
#define ZSTD_getSeqStore                         \
GBZSTD (ZSTD_getSeqStore)
#define ZSTD_initCStream                         \
GBZSTD (ZSTD_initCStream)
#define ZSTD_initCStream_advanced                \
GBZSTD (ZSTD_initCStream_advanced)
#define ZSTD_initCStream_internal                \
GBZSTD (ZSTD_initCStream_internal)
#define ZSTD_initCStream_srcSize                 \
GBZSTD (ZSTD_initCStream_srcSize)
#define ZSTD_initCStream_usingCDict              \
GBZSTD (ZSTD_initCStream_usingCDict)
#define ZSTD_initCStream_usingCDict_advanced     \
GBZSTD (ZSTD_initCStream_usingCDict_advanced)
#define ZSTD_initCStream_usingDict               \
GBZSTD (ZSTD_initCStream_usingDict)
#define ZSTD_initDStream                         \
GBZSTD (ZSTD_initDStream)
#define ZSTD_initDStream_usingDDict              \
GBZSTD (ZSTD_initDStream_usingDDict)
#define ZSTD_initDStream_usingDict               \
GBZSTD (ZSTD_initDStream_usingDict)
#define ZSTD_initStaticCCtx                      \
GBZSTD (ZSTD_initStaticCCtx)
#define ZSTD_initStaticCDict                     \
GBZSTD (ZSTD_initStaticCDict)
#define ZSTD_initStaticCStream                   \
GBZSTD (ZSTD_initStaticCStream)
#define ZSTD_initStaticDCtx                      \
GBZSTD (ZSTD_initStaticDCtx)
#define ZSTD_initStaticDDict                     \
GBZSTD (ZSTD_initStaticDDict)
#define ZSTD_initStaticDStream                   \
GBZSTD (ZSTD_initStaticDStream)
#define ZSTD_insertAndFindFirstIndex             \
GBZSTD (ZSTD_insertAndFindFirstIndex)
#define ZSTD_insertBlock                         \
GBZSTD (ZSTD_insertBlock)
#define ZSTD_invalidateRepCodes                  \
GBZSTD (ZSTD_invalidateRepCodes)
// #define ZSTD_isError                          \
// GBZSTD (ZSTD_isError)
#define ZSTD_isFrame                             \
GBZSTD (ZSTD_isFrame)
#define ZSTD_isSkippableFrame                    \
GBZSTD (ZSTD_isSkippableFrame)
#define ZSTD_ldm_adjustParameters                \
GBZSTD (ZSTD_ldm_adjustParameters)
#define ZSTD_ldm_blockCompress                   \
GBZSTD (ZSTD_ldm_blockCompress)
#define ZSTD_ldm_fillHashTable                   \
GBZSTD (ZSTD_ldm_fillHashTable)
#define ZSTD_ldm_generateSequences               \
GBZSTD (ZSTD_ldm_generateSequences)
#define ZSTD_ldm_getMaxNbSeq                     \
GBZSTD (ZSTD_ldm_getMaxNbSeq)
#define ZSTD_ldm_getTableSize                    \
GBZSTD (ZSTD_ldm_getTableSize)
#define ZSTD_ldm_skipRawSeqStoreBytes            \
GBZSTD (ZSTD_ldm_skipRawSeqStoreBytes)
#define ZSTD_ldm_skipSequences                   \
GBZSTD (ZSTD_ldm_skipSequences)
#define ZSTD_loadCEntropy                        \
GBZSTD (ZSTD_loadCEntropy)
#define ZSTD_loadDEntropy                        \
GBZSTD (ZSTD_loadDEntropy)
#define ZSTD_maxCLevel                           \
GBZSTD (ZSTD_maxCLevel)
#define ZSTD_mergeBlockDelimiters                \
GBZSTD (ZSTD_mergeBlockDelimiters)
#define ZSTD_minCLevel                           \
GBZSTD (ZSTD_minCLevel)
#define ZSTDMT_compressStream_generic            \
GBZSTD (ZSTDMT_compressStream_generic)
#define ZSTDMT_createCCtx_advanced               \
GBZSTD (ZSTDMT_createCCtx_advanced)
#define ZSTDMT_freeCCtx                          \
GBZSTD (ZSTDMT_freeCCtx)
#define ZSTDMT_getFrameProgression               \
GBZSTD (ZSTDMT_getFrameProgression)
#define ZSTDMT_initCStream_internal              \
GBZSTD (ZSTDMT_initCStream_internal)
#define ZSTDMT_nextInputSizeHint                 \
GBZSTD (ZSTDMT_nextInputSizeHint)
#define ZSTDMT_sizeof_CCtx                       \
GBZSTD (ZSTDMT_sizeof_CCtx)
#define ZSTDMT_toFlushNow                        \
GBZSTD (ZSTDMT_toFlushNow)
#define ZSTDMT_updateCParams_whileCompressing    \
GBZSTD (ZSTDMT_updateCParams_whileCompressing)
#define ZSTD_nextInputType                       \
GBZSTD (ZSTD_nextInputType)
#define ZSTD_nextSrcSizeToDecompress             \
GBZSTD (ZSTD_nextSrcSizeToDecompress)
#define ZSTD_noCompressLiterals                  \
GBZSTD (ZSTD_noCompressLiterals)
#define ZSTD_readSkippableFrame                  \
GBZSTD (ZSTD_readSkippableFrame)
#define ZSTD_referenceExternalSequences          \
GBZSTD (ZSTD_referenceExternalSequences)
#define ZSTD_reset_compressedBlockState          \
GBZSTD (ZSTD_reset_compressedBlockState)
#define ZSTD_resetCStream                        \
GBZSTD (ZSTD_resetCStream)
#define ZSTD_resetDStream                        \
GBZSTD (ZSTD_resetDStream)
#define ZSTD_resetSeqStore                       \
GBZSTD (ZSTD_resetSeqStore)
#define ZSTD_row_update                          \
GBZSTD (ZSTD_row_update)
#define ZSTD_selectBlockCompressor               \
GBZSTD (ZSTD_selectBlockCompressor)
#define ZSTD_selectEncodingType                  \
GBZSTD (ZSTD_selectEncodingType)
#define ZSTD_seqToCodes                          \
GBZSTD (ZSTD_seqToCodes)
#define ZSTD_sizeof_CCtx                         \
GBZSTD (ZSTD_sizeof_CCtx)
#define ZSTD_sizeof_CDict                        \
GBZSTD (ZSTD_sizeof_CDict)
#define ZSTD_sizeof_CStream                      \
GBZSTD (ZSTD_sizeof_CStream)
#define ZSTD_sizeof_DCtx                         \
GBZSTD (ZSTD_sizeof_DCtx)
#define ZSTD_sizeof_DDict                        \
GBZSTD (ZSTD_sizeof_DDict)
#define ZSTD_sizeof_DStream                      \
GBZSTD (ZSTD_sizeof_DStream)
#define ZSTD_toFlushNow                          \
GBZSTD (ZSTD_toFlushNow)
#define ZSTD_trace_compress_begin                \
GBZSTD (ZSTD_trace_compress_begin)
#define ZSTD_trace_compress_end                  \
GBZSTD (ZSTD_trace_compress_end)
#define ZSTD_trace_decompress_begin              \
GBZSTD (ZSTD_trace_decompress_begin)
#define ZSTD_trace_decompress_end                \
GBZSTD (ZSTD_trace_decompress_end)
#define ZSTD_updateTree                          \
GBZSTD (ZSTD_updateTree)
#define ZSTD_versionNumber                       \
GBZSTD (ZSTD_versionNumber)
#define ZSTD_versionString                       \
GBZSTD (ZSTD_versionString)
#define ZSTD_writeLastEmptyBlock                 \
GBZSTD (ZSTD_writeLastEmptyBlock)
#define ZSTD_writeSkippableFrame                 \
GBZSTD (ZSTD_writeSkippableFrame)
#define ZSTD_XXH32                               \
GBZSTD (ZSTD_XXH32)
#define ZSTD_XXH32_canonicalFromHash             \
GBZSTD (ZSTD_XXH32_canonicalFromHash)
#define ZSTD_XXH32_copyState                     \
GBZSTD (ZSTD_XXH32_copyState)
#define ZSTD_XXH32_createState                   \
GBZSTD (ZSTD_XXH32_createState)
#define ZSTD_XXH32_digest                        \
GBZSTD (ZSTD_XXH32_digest)
#define ZSTD_XXH32_freeState                     \
GBZSTD (ZSTD_XXH32_freeState)
#define ZSTD_XXH32_hashFromCanonical             \
GBZSTD (ZSTD_XXH32_hashFromCanonical)
#define ZSTD_XXH32_reset                         \
GBZSTD (ZSTD_XXH32_reset)
#define ZSTD_XXH32_update                        \
GBZSTD (ZSTD_XXH32_update)
#define ZSTD_XXH64                               \
GBZSTD (ZSTD_XXH64)
#define ZSTD_XXH64_canonicalFromHash             \
GBZSTD (ZSTD_XXH64_canonicalFromHash)
#define ZSTD_XXH64_copyState                     \
GBZSTD (ZSTD_XXH64_copyState)
#define ZSTD_XXH64_createState                   \
GBZSTD (ZSTD_XXH64_createState)
#define ZSTD_XXH64_digest                        \
GBZSTD (ZSTD_XXH64_digest)
#define ZSTD_XXH64_freeState                     \
GBZSTD (ZSTD_XXH64_freeState)
#define ZSTD_XXH64_hashFromCanonical             \
GBZSTD (ZSTD_XXH64_hashFromCanonical)
#define ZSTD_XXH64_reset                         \
GBZSTD (ZSTD_XXH64_reset)
#define ZSTD_XXH64_update                        \
GBZSTD (ZSTD_XXH64_update)
#define ZSTD_XXH_versionNumber                   \
GBZSTD (ZSTD_XXH_versionNumber)

//------------------------------------------------------------------------------
// disable ZSTD deprecation warnings and include all ZSTD definitions  
//------------------------------------------------------------------------------

// GraphBLAS does not use deprecated functions, but the warnings pop up anyway
// when GraphBLAS is built, so silence them with this #define:
#define ZSTD_DISABLE_DEPRECATE_WARNINGS

// do not use multithreading in ZSTD itself.  GraphBLAS does the parallelism.
#undef ZSTD_MULTITHREAD

// do not use asm
#define ZSTD_DISABLE_ASM
#include "zstd.h"
#endif

