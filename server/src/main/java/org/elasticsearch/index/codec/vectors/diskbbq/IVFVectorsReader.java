/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.index.codec.vectors.GenericFlatVectorReaders;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsReader;
import org.elasticsearch.search.vectors.ESAcceptDocs;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;
import org.elasticsearch.search.vectors.IncrementalDeduplicationFilter;
import org.elasticsearch.search.vectors.IncrementalFilterIterator;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.CENTROID_EXTENSION;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.CLUSTER_EXTENSION;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.DYNAMIC_VISIT_RATIO;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.VERSION_DIRECT_IO;

/**
 * Reader for IVF vectors. This reader is used to read the IVF vectors from the index.
 */
public abstract class IVFVectorsReader extends KnnVectorsReader {

    private final IndexInput ivfCentroids, ivfClusters;
    private final SegmentReadState state;
    private final FieldInfos fieldInfos;
    protected final IntObjectHashMap<FieldEntry> fields;
    private final GenericFlatVectorReaders genericReaders;

    @SuppressWarnings("this-escape")
    protected IVFVectorsReader(SegmentReadState state, GenericFlatVectorReaders.LoadFlatVectorsReader loadReader) throws IOException {
        this.state = state;
        this.fieldInfos = state.fieldInfos;
        this.fields = new IntObjectHashMap<>();
        this.genericReaders = new GenericFlatVectorReaders();
        String meta = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            ES920DiskBBQVectorsFormat.IVF_META_EXTENSION
        );

        int versionMeta = -1;
        try (ChecksumIndexInput ivfMeta = state.directory.openChecksumInput(meta)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    ivfMeta,
                    ES920DiskBBQVectorsFormat.NAME,
                    ES920DiskBBQVectorsFormat.VERSION_START,
                    ES920DiskBBQVectorsFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(ivfMeta, versionMeta, genericReaders, loadReader);
            } catch (Throwable exception) {
                priorE = exception;
            } finally {
                CodecUtil.checkFooter(ivfMeta, priorE);
            }
            ivfCentroids = openDataInput(state, versionMeta, CENTROID_EXTENSION, ES920DiskBBQVectorsFormat.NAME, state.context);
            ivfClusters = openDataInput(state, versionMeta, CLUSTER_EXTENSION, ES920DiskBBQVectorsFormat.NAME, state.context);
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(this);
            throw t;
        }
    }

    public abstract CentroidIterator getCentroidIterator(
        FieldInfo fieldInfo,
        int numCentroids,
        IndexInput centroids,
        float[] target,
        IndexInput postingListSlice,
        AcceptDocs acceptDocs,
        float approximateCost,
        FloatVectorValues values,
        float visitRatio
    ) throws IOException;

    private static IndexInput openDataInput(
        SegmentReadState state,
        int versionMeta,
        String fileExtension,
        String codecName,
        IOContext context
    ) throws IOException {
        final String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
        final IndexInput in = state.directory.openInput(fileName, context);
        try {
            final int versionVectorData = CodecUtil.checkIndexHeader(
                in,
                codecName,
                ES920DiskBBQVectorsFormat.VERSION_START,
                ES920DiskBBQVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (versionMeta != versionVectorData) {
                throw new CorruptIndexException(
                    "Format versions mismatch: meta=" + versionMeta + ", " + codecName + "=" + versionVectorData,
                    in
                );
            }
            CodecUtil.retrieveChecksum(in);
            return in;
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(in);
            throw t;
        }
    }

    private void readFields(
        ChecksumIndexInput meta,
        int versionMeta,
        GenericFlatVectorReaders genericFields,
        GenericFlatVectorReaders.LoadFlatVectorsReader loadReader
    ) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            final FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }

            FieldEntry fieldEntry = readField(meta, info, versionMeta);
            genericFields.loadField(fieldNumber, fieldEntry, loadReader);

            fields.put(info.number, fieldEntry);
        }
    }

    private FieldEntry readField(IndexInput input, FieldInfo info, int versionMeta) throws IOException {
        final String rawVectorFormat = input.readString();
        final boolean useDirectIOReads = versionMeta >= VERSION_DIRECT_IO && input.readByte() == 1;
        final VectorEncoding vectorEncoding = readVectorEncoding(input);
        final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
        if (similarityFunction != info.getVectorSimilarityFunction()) {
            throw new IllegalStateException(
                "Inconsistent vector similarity function for field=\""
                    + info.name
                    + "\"; "
                    + similarityFunction
                    + " != "
                    + info.getVectorSimilarityFunction()
            );
        }
        final int numCentroids = input.readInt();
        final long centroidOffset = input.readLong();
        final long centroidLength = input.readLong();
        final float[] globalCentroid = new float[info.getVectorDimension()];
        long postingListOffset = -1;
        long postingListLength = -1;
        float globalCentroidDp = 0;
        if (centroidLength > 0) {
            postingListOffset = input.readLong();
            postingListLength = input.readLong();
            input.readFloats(globalCentroid, 0, globalCentroid.length);
            globalCentroidDp = Float.intBitsToFloat(input.readInt());
        }
        return doReadField(
            input,
            rawVectorFormat,
            useDirectIOReads,
            similarityFunction,
            vectorEncoding,
            numCentroids,
            centroidOffset,
            centroidLength,
            postingListOffset,
            postingListLength,
            globalCentroid,
            globalCentroidDp
        );
    }

    protected abstract FieldEntry doReadField(
        IndexInput input,
        String rawVectorFormat,
        boolean useDirectIOReads,
        VectorSimilarityFunction similarityFunction,
        VectorEncoding vectorEncoding,
        int numCentroids,
        long centroidOffset,
        long centroidLength,
        long postingListOffset,
        long postingListLength,
        float[] globalCentroid,
        float globalCentroidDp
    ) throws IOException;

    private static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
        final int i = input.readInt();
        if (i < 0 || i >= SIMILARITY_FUNCTIONS.size()) {
            throw new IllegalArgumentException("invalid distance function: " + i);
        }
        return SIMILARITY_FUNCTIONS.get(i);
    }

    private static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
        final int encodingId = input.readInt();
        if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
            throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
        }
        return VectorEncoding.values()[encodingId];
    }

    @Override
    public final void checkIntegrity() throws IOException {
        for (var reader : genericReaders.allReaders()) {
            reader.checkIntegrity();
        }
        CodecUtil.checksumEntireFile(ivfCentroids);
        CodecUtil.checksumEntireFile(ivfClusters);
    }

    private FlatVectorsReader getReaderForField(String field) {
        FieldInfo info = fieldInfos.fieldInfo(field);
        if (info == null) throw new IllegalArgumentException("Could not find field [" + field + "]");
        return genericReaders.getReaderForField(info.number);
    }

    @Override
    public final FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return getReaderForField(field).getFloatVectorValues(field);
    }

    @Override
    public final ByteVectorValues getByteVectorValues(String field) throws IOException {
        return getReaderForField(field).getByteVectorValues(field);
    }

    @Override
    public final void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32) == false) {
            getReaderForField(field).search(field, target, knnCollector, acceptDocs);
            return;
        }
        if (fieldInfo.getVectorDimension() != target.length) {
            throw new IllegalArgumentException(
                "vector query dimension: " + target.length + " differs from field dimension: " + fieldInfo.getVectorDimension()
            );
        }
        final ESAcceptDocs esAcceptDocs;
        if (acceptDocs instanceof ESAcceptDocs) {
            esAcceptDocs = (ESAcceptDocs) acceptDocs;
        } else {
            esAcceptDocs = null;
        }
        FloatVectorValues values = getReaderForField(field).getFloatVectorValues(field);
        int numVectors = values.size();
        // TODO returning cost 0 in ESAcceptDocs.ESAcceptDocsAll feels wrong? cost is related to the number of matching documents?
        float approximateCost = (float) (esAcceptDocs == null ? acceptDocs.cost()
            : esAcceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll ? numVectors
            : esAcceptDocs.approximateCost());
        float percentFiltered = Math.max(0f, Math.min(1f, approximateCost / numVectors));
        float visitRatio = DYNAMIC_VISIT_RATIO;
        // Search strategy may be null if this is being called from checkIndex (e.g. from a test)
        if (knnCollector.getSearchStrategy() instanceof IVFKnnSearchStrategy ivfSearchStrategy) {
            visitRatio = ivfSearchStrategy.getVisitRatio();
        }

        FieldEntry entry = fields.get(fieldInfo.number);
        if (visitRatio == DYNAMIC_VISIT_RATIO) {
            // empirically based, and a good dynamic to get decent recall while scaling a la "efSearch"
            // scaling by the number of vectors vs. the nearest neighbors requested
            // not perfect, but a comparative heuristic.
            // TODO: we might want to consider the density of the centroids as experiments shows that for fewer vectors per centroid,
            // the least vectors we need to score to get a good recall.
            float estimated = Math.round(Math.log10(numVectors) * Math.log10(numVectors) * (knnCollector.k()));
            // clip so we visit at least one vector
            visitRatio = estimated / numVectors;
        }
        // we account for soar vectors here. We can potentially visit a vector twice so we multiply by 2 here.
        long maxVectorVisited = (long) (2.0 * visitRatio * numVectors);
        IndexInput postListSlice = entry.postingListSlice(ivfClusters);
        CentroidIterator centroidPrefetchingIterator = getCentroidIterator(
            fieldInfo,
            entry.numCentroids,
            entry.centroidSlice(ivfCentroids),
            target,
            postListSlice,
            acceptDocs,
            approximateCost,
            values,
            visitRatio
        );

        // TODO: Enable parallel posting list evaluation conditionally
        // For now, disabled by default - requires all readers to support batch operations
        // The issue is that if we try parallel and it fails, the centroidIterator is already
        // partially consumed and we can't rewind it for sequential processing
        boolean useParallelEvaluation = this instanceof ESNextDiskBBQVectorsReader;

        if (useParallelEvaluation) {
            try {
                // Try to use parallel evaluation
                // If the PostingVisitor doesn't support batch operations, this will throw
                // UnsupportedOperationException and we'll fall back to sequential
                searchWithParallelPostingLists(
                    fieldInfo,
                    postListSlice,
                    target,
                    acceptDocs, // Pass full AcceptDocs for iterator access
                    centroidPrefetchingIterator,
                    knnCollector,
                    maxVectorVisited,
                    state.segmentInfo.maxDoc(),
                    numVectors,
                    percentFiltered
                );
                return; // Success - exit early
            } catch (UnsupportedOperationException e) {
                // Batch operations not supported by this visitor, fall back to sequential
                // Continue with sequential implementation below
            }
        }
        Bits acceptDocsBits = acceptDocs.bits();
        PostingVisitor scorer = getPostingVisitor(fieldInfo, postListSlice, target, acceptDocsBits);
        long expectedDocs = 0;
        long actualDocs = 0;
        // initially we visit only the "centroids to search"
        // Note, numCollected is doing the bare minimum here.
        // TODO do we need to handle nested doc counts similarly to how we handle
        // filtering? E.g. keep exploring until we hit an expected number of parent documents vs. child vectors?
        while (centroidPrefetchingIterator.hasNext()
            && (maxVectorVisited > expectedDocs || knnCollector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY)) {
            // todo do we actually need to know the score???
            CentroidMeta offsetAndLength = centroidPrefetchingIterator.nextCentroidMeta();
            // todo do we need direct access to the raw centroid???, this is used for quantizing, maybe hydrating and quantizing
            // is enough?
            expectedDocs += scorer.resetPostingsScorer(offsetAndLength.offset());
            actualDocs += scorer.visit(knnCollector);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
        }
        if (acceptDocsBits != null) {
            // TODO Adjust the value here when using centroid filtering
            float unfilteredRatioVisited = (float) expectedDocs / numVectors;
            int filteredVectors = (int) Math.ceil(numVectors * percentFiltered);
            float expectedScored = Math.min(2 * filteredVectors * unfilteredRatioVisited, expectedDocs / 2f);
            while (centroidPrefetchingIterator.hasNext() && (actualDocs < expectedScored || actualDocs < knnCollector.k())) {
                CentroidMeta offsetAndLength = centroidPrefetchingIterator.nextCentroidMeta();
                scorer.resetPostingsScorer(offsetAndLength.offset());
                actualDocs += scorer.visit(knnCollector);
                if (knnCollector.getSearchStrategy() != null) {
                    knnCollector.getSearchStrategy().nextVectorsBlock();
                }
            }
        }
    }

    /**
     * Searches using parallel posting list evaluation with multi-way merge.
     * This approach processes multiple posting lists simultaneously, merging
     * documents in doc ID order to enable efficient deduplication and early termination.
     *
     * @param fieldInfo the field info for the vector field
     * @param postingListInput the input for reading posting lists
     * @param target the query vector
     * @param centroidIterator iterator providing posting list offsets
     * @param knnCollector the collector for results
     * @param maxVectorVisited maximum number of vectors to visit
     * @param maxDoc maximum document ID in the segment
     * @throws IOException if an I/O error occurs
     */
    private void searchWithParallelPostingLists(
        FieldInfo fieldInfo,
        IndexInput postingListInput,
        float[] target,
        AcceptDocs acceptDocs,
        CentroidIterator centroidIterator,
        KnnCollector knnCollector,
        long maxVectorVisited,
        int maxDoc,
        int numVectors,
        float percentFiltered
    ) throws IOException {
        // Strategy: Load doc IDs from multiple posting lists, filter them, then batch score
        // This separates cheap operations (doc ID loading + filtering) from expensive operations (vector scoring)

        final int BATCH_SIZE = 1000; // Number of posting lists to process in parallel
        final IncrementalDeduplicationFilter deduplicationFilter = new IncrementalDeduplicationFilter(10_000);

        // Initialize incremental filter iterator from acceptDocs
        // Try to get an iterator for efficient filtering, but fall back to bits() if iterator is not available
        DocIdSetIterator filterIterator = acceptDocs == null ? null : acceptDocs.iterator();
        final IncrementalFilterIterator incrementalFilterIterator = new IncrementalFilterIterator(filterIterator, maxDoc);

        long expectedDocs = 0;   // Total vectors in posting lists we've loaded
        long actualDocs = 0;     // Docs actually scored (after filtering)
        PostingVisitor visitor = getPostingVisitor(fieldInfo, postingListInput, target, null);

        while (centroidIterator.hasNext()
            && (maxVectorVisited > expectedDocs || knnCollector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY)) {

            // load BATCH_SIZE postings lists
            int batchCount = 0;
            int[] minDocIds = new int[BATCH_SIZE];
            long[] postingListSizes = new long[BATCH_SIZE];
            long[] offsets = new long[BATCH_SIZE];

            // while there are more centroids to explore and we haven't reached our batch_size
            // let's load another set of centroids
            while (centroidIterator.hasNext() && batchCount < BATCH_SIZE) {
                CentroidMeta centroid = centroidIterator.nextCentroidMeta();
                // Don't pass acceptDocs to visitor - we handle filtering via IncrementalFilterIterator
                // Reset visitor to this posting list and get its size
                long size = visitor.resetPostingsScorer(centroid.offset());
                if (size > 0) {
                    int minDocId = visitor.peekFirstDocId();
                    minDocIds[batchCount] = minDocId;
                    offsets[batchCount] = centroid.offset;
                    postingListSizes[batchCount] = size;  // Store size, don't increment expectedDocs yet
                    batchCount++;
                }
            }

            if (batchCount == 0) {
                break;
            }

            // Sort posting lists by min doc ID to enable efficient multi-way merge
            Integer[] indices = new Integer[batchCount];
            for (int i = 0; i < batchCount; i++) {
                indices[i] = i;
            }
            Arrays.sort(indices, 0, batchCount, (a, b) -> Integer.compare(minDocIds[a], minDocIds[b]));

            // Step 3: Process posting lists in min doc ID order
            for (int idx = 0; idx < batchCount; idx++) {
                // Check termination condition BEFORE processing each posting list
                // This matches the sequential behavior where condition is checked before each centroid
                if (maxVectorVisited <= expectedDocs && knnCollector.minCompetitiveSimilarity() != Float.NEGATIVE_INFINITY) {
                    break;
                }

                int i = indices[idx];

                // Increment expectedDocs just before processing (matching sequential behavior)
                expectedDocs += postingListSizes[i];
                visitor.resetPostingsScorer(offsets[i]);
                // Visit all docs from this posting list, applying filters
                actualDocs += visitor.visitFiltered(knnCollector, incrementalFilterIterator, deduplicationFilter);

                if (knnCollector.getSearchStrategy() != null) {
                    knnCollector.getSearchStrategy().nextVectorsBlock();
                }
            }
        }
        if (acceptDocs != null) {
            // TODO Adjust the value here when using centroid filtering
            float unfilteredRatioVisited = (float) expectedDocs / numVectors;
            int filteredVectors = (int) Math.ceil(numVectors * percentFiltered);
            float expectedScored = Math.min(2 * filteredVectors * unfilteredRatioVisited, expectedDocs / 2f);
            while (centroidIterator.hasNext() && (actualDocs < expectedScored || actualDocs < knnCollector.k())) {
                CentroidMeta offsetAndLength = centroidIterator.nextCentroidMeta();
                visitor.resetPostingsScorer(offsetAndLength.offset());
                actualDocs += visitor.visitFiltered(knnCollector, incrementalFilterIterator, deduplicationFilter);
                if (knnCollector.getSearchStrategy() != null) {
                    knnCollector.getSearchStrategy().nextVectorsBlock();
                }
            }
        }
        // // Phase 2: Filter compensation - if filtering is active, visit more centroids
        // if (acceptDocs != null) {
        // float unfilteredRatioVisited = (float) expectedDocs / numVectors;
        // int filteredVectors = (int) Math.ceil(numVectors * percentFiltered);
        // float expectedScored = Math.min(2 * filteredVectors * unfilteredRatioVisited, expectedDocs / 2f);
        // while (centroidIterator.hasNext() && (actualDocs < expectedScored || actualDocs < knnCollector.k())) {
        // // Load next batch
        // int batchCount = 0;
        // PostingVisitor[] visitors = new PostingVisitor[BATCH_SIZE];
        // int[] minDocIds = new int[BATCH_SIZE];
        // long[] postingListSizes = new long[BATCH_SIZE];
        //
        // while(centroidIterator.hasNext() && batchCount < BATCH_SIZE) {
        // CentroidMeta centroid = centroidIterator.nextCentroidMeta();
        // // Don't pass acceptDocs to visitor - we handle filtering via IncrementalFilterIterator
        // PostingVisitor visitor = getPostingVisitor(fieldInfo, postingListInput, target, null);
        //
        // long size = visitor.resetPostingsScorer(centroid.offset());
        // if (size > 0) {
        // int minDocId = visitor.peekFirstDocId();
        // visitors[batchCount] = visitor;
        // minDocIds[batchCount] = minDocId;
        // postingListSizes[batchCount] = size; // Store size, don't increment expectedDocs yet
        // batchCount++;
        // }
        // }
        //
        // if (batchCount == 0) {
        // break;
        // }
        //
        // // Sort posting lists by min doc ID
        // Integer[] indices = new Integer[batchCount];
        // for (int i = 0; i < batchCount; i++) {
        // indices[i] = i;
        // }
        // java.util.Arrays.sort(indices, 0, batchCount, (a, b) -> Integer.compare(minDocIds[a], minDocIds[b]));
        //
        // // Process batch in min doc ID order
        // for (int idx = 0; idx < batchCount; idx++) {
        // // Check if we've scored enough docs before processing next posting list
        // if (actualDocs >= expectedScored && actualDocs >= knnCollector.k()) {
        // return;
        // }
        //
        // int i = indices[idx];
        //
        // // Increment expectedDocs just before processing (matching sequential behavior)
        // expectedDocs += postingListSizes[i];
        //
        // int scored = visitors[i].visitFiltered(knnCollector, incrementalFilterIterator, deduplicationFilter);
        // actualDocs += scored;
        //
        // if (knnCollector.getSearchStrategy() != null) {
        // knnCollector.getSearchStrategy().nextVectorsBlock();
        // }
        //
        // // Check for early termination after each posting list
        // if (knnCollector.earlyTerminated()) {
        // return;
        // }
        // }
        // }
        // }
    }

    @Override
    public final void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        final ByteVectorValues values = getReaderForField(field).getByteVectorValues(field);
        for (int i = 0; i < values.size(); i++) {
            final float score = fieldInfo.getVectorSimilarityFunction().compare(target, values.vectorValue(i));
            knnCollector.collect(values.ordToDoc(i), score);
            if (knnCollector.earlyTerminated()) {
                return;
            }
        }
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        var raw = getReaderForField(fieldInfo.name).getOffHeapByteSize(fieldInfo);
        FieldEntry fe = fields.get(fieldInfo.number);
        if (fe == null) {
            assert fieldInfo.getVectorEncoding() == VectorEncoding.BYTE;
            return raw;
        }

        var centroidsClusters = Map.of(CENTROID_EXTENSION, fe.centroidLength, CLUSTER_EXTENSION, fe.postingListLength);
        return KnnVectorsReader.mergeOffHeapByteSizeMaps(raw, centroidsClusters);
    }

    @Override
    public void close() throws IOException {
        List<Closeable> closeables = new ArrayList<>(genericReaders.allReaders());
        Collections.addAll(closeables, ivfCentroids, ivfClusters);
        IOUtils.close(closeables);
    }

    protected static class FieldEntry implements GenericFlatVectorReaders.Field {
        protected final String rawVectorFormatName;
        protected final boolean useDirectIOReads;
        protected final VectorSimilarityFunction similarityFunction;
        protected final VectorEncoding vectorEncoding;
        protected final int numCentroids;
        protected final long centroidOffset;
        protected final long centroidLength;
        protected final long postingListOffset;
        protected final long postingListLength;
        protected final float[] globalCentroid;
        protected final float globalCentroidDp;
        protected final int bulkSize;

        protected FieldEntry(
            String rawVectorFormatName,
            boolean useDirectIOReads,
            VectorSimilarityFunction similarityFunction,
            VectorEncoding vectorEncoding,
            int numCentroids,
            long centroidOffset,
            long centroidLength,
            long postingListOffset,
            long postingListLength,
            float[] globalCentroid,
            float globalCentroidDp,
            int bulkSize
        ) {
            this.rawVectorFormatName = rawVectorFormatName;
            this.useDirectIOReads = useDirectIOReads;
            this.similarityFunction = similarityFunction;
            this.vectorEncoding = vectorEncoding;
            this.numCentroids = numCentroids;
            this.centroidOffset = centroidOffset;
            this.centroidLength = centroidLength;
            this.postingListOffset = postingListOffset;
            this.postingListLength = postingListLength;
            this.globalCentroid = globalCentroid;
            this.globalCentroidDp = globalCentroidDp;
            this.bulkSize = bulkSize;
        }

        @Override
        public String rawVectorFormatName() {
            return rawVectorFormatName;
        }

        @Override
        public boolean useDirectIOReads() {
            return useDirectIOReads;
        }

        public int numCentroids() {
            return numCentroids;
        }

        public float[] globalCentroid() {
            return globalCentroid;
        }

        public float globalCentroidDp() {
            return globalCentroidDp;
        }

        public VectorSimilarityFunction similarityFunction() {
            return similarityFunction;
        }

        public IndexInput centroidSlice(IndexInput centroidFile) throws IOException {
            return centroidFile.slice("centroids", centroidOffset, centroidLength);
        }

        public IndexInput postingListSlice(IndexInput postingListFile) throws IOException {
            return postingListFile.slice("postingLists", postingListOffset, postingListLength);
        }

        public int getBulkSize() {
            return bulkSize;
        }
    }

    public abstract PostingVisitor getPostingVisitor(FieldInfo fieldInfo, IndexInput postingsLists, float[] target, Bits needsScoring)
        throws IOException;

    public record CentroidMeta(long offset, long length, int centroidOrdinal) {}

    public interface CentroidIterator {
        boolean hasNext();

        CentroidMeta nextCentroidMeta() throws IOException;
    }

    public interface PostingVisitor {
        /** returns the number of documents in the posting list */
        int resetPostingsScorer(long offset) throws IOException;

        /** returns the number of scored documents */
        int visit(KnnCollector collector) throws IOException;

        /**
         * Checks if there are more batches to read from the current posting list.
         * Must be called after {@link #resetPostingsScorer(long)}.
         *
         * @return true if more batches are available
         */
        default boolean hasNextBatch() {
            throw new UnsupportedOperationException("Batch-level operations not supported by this visitor");
        }

        /**
         * Reads the next batch of document IDs (without scoring).
         * Returns the document IDs in ascending order within the batch.
         * Must be called after {@link #resetPostingsScorer(long)}.
         *
         * @param docIds array to store the document IDs (must be at least 32 elements)
         * @return the number of document IDs read (0 if no more batches)
         * @throws IOException if an I/O error occurs
         */
        default int readNextBatch(int[] docIds) throws IOException {
            throw new UnsupportedOperationException("Batch-level operations not supported by this visitor");
        }

        /**
         * Returns the minimum document ID in the most recently read batch.
         * Valid after calling {@link #readNextBatch(int[])}.
         *
         * @return the minimum document ID in the current batch, or Integer.MAX_VALUE if no batch loaded
         */
        default int getMinDocInCurrentBatch() {
            return Integer.MAX_VALUE;
        }

        /**
         * Scores documents from the current batch and collects them.
         * Only scores documents whose IDs are in the provided array.
         * Must be called after {@link #readNextBatch(int[])}.
         *
         * @param docIds the document IDs to score (subset of batch)
         * @param count the number of valid document IDs in the array
         * @param collector the collector to receive scored documents
         * @return the number of documents actually scored
         * @throws IOException if an I/O error occurs
         */
        default int scoreCurrentBatch(int[] docIds, int count, KnnCollector collector) throws IOException {
            throw new UnsupportedOperationException("Batch-level operations not supported by this visitor");
        }

        /**
         * Scores ALL documents in batch using POST-FILTERING approach.
         * Scores all docs first (bulk efficiency), then applies filter + dedup to results.
         * Must be called after {@link #readNextBatch(int[])}.
         *
         * @param docIds all document IDs in the batch
         * @param count the number of document IDs
         * @param filterIterator incremental filter iterator for checking filter
         * @param deduplicationFilter deduplication filter
         * @param collector the collector to receive scored documents
         * @return the number of documents scored (before filtering)
         * @throws IOException if an I/O error occurs
         */
        default int scoreBulkWithPostFiltering(
            int[] docIds,
            int count,
            IncrementalFilterIterator filterIterator,
            IncrementalDeduplicationFilter deduplicationFilter,
            KnnCollector collector
        ) throws IOException {
            throw new UnsupportedOperationException("Post-filtering batch operations not supported by this visitor");
        }

        /**
         * Peeks at the first document ID in the posting list without advancing the reader.
         * This is used to determine the minimum doc ID for sorting posting lists.
         * Must be called after {@link #resetPostingsScorer(long)}.
         *
         * @return the first document ID in the posting list
         * @throws IOException if an I/O error occurs
         */
        default int peekFirstDocId() throws IOException {
            throw new UnsupportedOperationException("peekFirstDocId not supported by this visitor");
        }

        /**
         * Visits all documents in the posting list, applying post-filtering.
         * This implements the score-then-filter strategy:
         * 1. Read and score all docs from the posting list in batches
         * 2. Apply acceptDocs filter and deduplication to scored results
         * 3. Collect only filtered docs
         *
         * Must be called after {@link #resetPostingsScorer(long)}.
         *
         * @param collector the collector to receive scored documents
         * @param deduplicationFilter filter to track and skip already-seen documents
         * @return the number of documents actually collected (after filtering)
         * @throws IOException if an I/O error occurs
         */
        default int visitFiltered(
            KnnCollector collector,
            IncrementalFilterIterator filterIterator,
            IncrementalDeduplicationFilter deduplicationFilter
        ) throws IOException {
            throw new UnsupportedOperationException("visitFiltered not supported by this visitor");
        }
    }
}
