/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.CheckJoinIndex;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.diskbbq.TestIvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.elasticsearch.index.engine.Engine;
import org.junit.After;
import org.junit.Before;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.LongAccumulator;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomBoolean;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomInt;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomIntBetween;

/**
 * Verifies that within-segment parallel scanning of IVF posting lists preserves the serial algorithm's results.
 *
 * <p>The parallel design guarantees the parallel run visits a superset of the postings the serial run visits (workers
 * claim ranked postings in order and the caller resumes the serial loop's exact termination conditions on the
 * remainder). Per-document scores are deterministic, so two invariants must hold on the same single-segment index:
 * <ul>
 *     <li>with a visit ratio of 1.0 both runs visit everything and results must be identical, up to permutations of
 *     docs with exactly tied scores (block-level pruning compares scores only, while queue ordering tiebreaks on doc
 *     id, so floor-propagation timing can reorder exact ties), and</li>
 *     <li>with a partial visit ratio the parallel k-th best score must be at least the serial one.</li>
 * </ul>
 *
 * <p>The parallel scan context is constructed directly (rather than through the {@code es.vectors.*} system
 * properties) so the parallel path is exercised regardless of the opt-in flag's default.
 */
public class IVFIntraSegmentParallelScanTests extends LuceneTestCase {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    private static final String FIELD = "field";
    private static final String SLICE_FIELD = "_slice";
    private static final String FILTER_FIELD = "_filter";
    private static final String FILTER_MATCH = "match";

    private ExecutorService executor;
    private TaskExecutor taskExecutor;
    private int workers;

    @Before
    public void setUpExecutor() {
        workers = randomIntBetween(2, 4);
        executor = Executors.newFixedThreadPool(workers);
        taskExecutor = new TaskExecutor(executor);
    }

    @After
    public void tearDownExecutor() {
        executor.shutdownNow();
    }

    public void testExhaustiveVisitParity() throws IOException {
        int numVectors = randomIntBetween(1500, 2500);
        int dims = randomIntBetween(16, 48);
        try (Directory dir = newDirectory()) {
            indexSingleSegment(dir, numVectors, dims, 0);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 50);
                    // visit ratio 1.0 covers every posting in both runs, so results must match (up to exact ties)
                    TopDocs serial = searchLeaf(leaf, query, k, 1.0f, null, null);
                    TopDocs parallel = searchLeaf(leaf, query, k, 1.0f, newParallelContext(null), null);
                    assertSameTopDocs(serial, parallel);
                    assertEquals(serial.totalHits.value(), parallel.totalHits.value());
                }
            }
        }
    }

    public void testPartialVisitSupersetDominance() throws IOException {
        int numVectors = randomIntBetween(3000, 5000);
        int dims = randomIntBetween(16, 48);
        try (Directory dir = newDirectory()) {
            indexSingleSegment(dir, numVectors, dims, 0);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    float visitRatio = randomFloat() * 0.15f + 0.05f;
                    TopDocs serial = searchLeaf(leaf, query, k, visitRatio, null, null);
                    TopDocs parallel = searchLeaf(leaf, query, k, visitRatio, newParallelContext(null), null);
                    assertEquals(serial.scoreDocs.length, parallel.scoreDocs.length);
                    // the parallel visit set is a superset of the serial one, so its k-th best cannot be worse
                    float serialMin = serial.scoreDocs[serial.scoreDocs.length - 1].score;
                    float parallelMin = parallel.scoreDocs[parallel.scoreDocs.length - 1].score;
                    assertTrue(
                        "parallel k-th score " + parallelMin + " must be >= serial k-th score " + serialMin,
                        parallelMin >= serialMin
                    );
                    assertTrue("parallel must visit at least as much", parallel.totalHits.value() >= serial.totalHits.value());
                }
            }
        }
    }

    /**
     * Exercises the filtered adaptive loop with a binding visit budget: the workers' aggregated expected/actual
     * counts feed the {@code expectedScored} math of the serial continuation, which no other test reaches with a
     * partial visit ratio.
     */
    public void testFilteredPartialVisitDominance() throws IOException {
        int numVectors = randomIntBetween(3000, 5000);
        int dims = randomIntBetween(16, 48);
        try (Directory dir = newDirectory()) {
            indexSingleSegment(dir, numVectors, dims, 0);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                int maxDoc = leaf.reader().maxDoc();
                FixedBitSet filter = new FixedBitSet(maxDoc);
                for (int doc = 0; doc < maxDoc; doc++) {
                    if (randomInt(9) < 4) {
                        filter.set(doc);
                    }
                }
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    float visitRatio = randomFloat() * 0.15f + 0.05f;
                    TopDocs serial = searchLeaf(leaf, query, k, visitRatio, null, filter);
                    TopDocs parallel = searchLeaf(leaf, query, k, visitRatio, newParallelContext(null), filter);
                    assertEquals(serial.scoreDocs.length, parallel.scoreDocs.length);
                    if (serial.scoreDocs.length > 0) {
                        float serialMin = serial.scoreDocs[serial.scoreDocs.length - 1].score;
                        float parallelMin = parallel.scoreDocs[parallel.scoreDocs.length - 1].score;
                        assertTrue(
                            "parallel k-th score " + parallelMin + " must be >= serial k-th score " + serialMin,
                            parallelMin >= serialMin
                        );
                    }
                }
            }
        }
    }

    public void testFilteredParity() throws IOException {
        int numVectors = randomIntBetween(2000, 3000);
        int dims = randomIntBetween(16, 48);
        try (Directory dir = newDirectory()) {
            indexSingleSegment(dir, numVectors, dims, 0);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                int maxDoc = leaf.reader().maxDoc();
                FixedBitSet filter = new FixedBitSet(maxDoc);
                for (int doc = 0; doc < maxDoc; doc++) {
                    if (randomInt(9) < 4) {
                        filter.set(doc);
                    }
                }
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    TopDocs serial = searchLeaf(leaf, query, k, 1.0f, null, filter);
                    TopDocs parallel = searchLeaf(leaf, query, k, 1.0f, newParallelContext(null), filter);
                    assertSameTopDocs(serial, parallel);
                }
            }
        }
    }

    public void testDeletesParity() throws IOException {
        int numVectors = randomIntBetween(2000, 3000);
        int dims = randomIntBetween(16, 48);
        try (Directory dir = newDirectory()) {
            indexSingleSegment(dir, numVectors, dims, numVectors / 5);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                Bits liveDocs = leaf.reader().getLiveDocs();
                assertNotNull("test expects deletes to produce live docs", liveDocs);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    TopDocs serial = searchLeaf(leaf, query, k, 1.0f, null, null);
                    TopDocs parallel = searchLeaf(leaf, query, k, 1.0f, newParallelContext(null), null);
                    assertSameTopDocs(serial, parallel);
                }
            }
        }
    }

    public void testCancellationIsObserved() throws IOException {
        int numVectors = randomIntBetween(1500, 2500);
        int dims = 16;
        try (Directory dir = newDirectory()) {
            indexSingleSegment(dir, numVectors, dims, 0);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                IVFParallelScanContext cancellingContext = newParallelContext(() -> { throw new RuntimeException("cancelled"); });
                RuntimeException e = expectThrows(
                    RuntimeException.class,
                    () -> searchLeaf(leaf, randomVector(dims), 10, 1.0f, cancellingContext, null)
                );
                assertTrue(e.getMessage(), e.getMessage() != null && e.getMessage().contains("cancelled"));
            }
        }
    }

    /** Filters and deletes together: the visitor sees a bitset masking both, unlike the individual tests above. */
    public void testFilteredWithDeletesParityAndDominance() throws IOException {
        int numVectors = randomIntBetween(2000, 3000);
        int dims = randomIntBetween(16, 48);
        try (Directory dir = newDirectory()) {
            indexSingleSegment(dir, numVectors, dims, numVectors / 5);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                assertNotNull("test expects deletes to produce live docs", leaf.reader().getLiveDocs());
                int maxDoc = leaf.reader().maxDoc();
                FixedBitSet filter = new FixedBitSet(maxDoc);
                for (int doc = 0; doc < maxDoc; doc++) {
                    if (randomInt(9) < 4) {
                        filter.set(doc);
                    }
                }
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    TopDocs serial = searchLeaf(leaf, query, k, 1.0f, null, filter);
                    TopDocs parallel = searchLeaf(leaf, query, k, 1.0f, newParallelContext(null), filter);
                    assertSameTopDocs(serial, parallel);

                    float visitRatio = randomFloat() * 0.15f + 0.05f;
                    serial = searchLeaf(leaf, query, k, visitRatio, null, filter);
                    parallel = searchLeaf(leaf, query, k, visitRatio, newParallelContext(null), filter);
                    assertKthScoreDominance(serial, parallel);
                }
            }
        }
    }

    /**
     * Diversified (parent/child) leaf searches: parallel workers use private diversifying collectors sharing the
     * leaf's parent bitset, and the merge into the leaf collector re-enforces the one-best-child-per-parent
     * invariant, so parity/dominance must hold exactly as for the plain collector.
     */
    public void testDiversifiedParityAndDominance() throws IOException {
        int numParents = randomIntBetween(150, 250);
        int dims = randomIntBetween(16, 48);
        try (Directory dir = newDirectory()) {
            indexParentChildSegments(dir, numParents, dims, 1);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReaderContext leaf = getOnlyLeaf(reader);
                BitSetProducer parentsFilter = parentFilter(reader);
                BitSet parentBitSet = parentsFilter.getBitSet(leaf);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    TopDocs serial = searchDiversifiedLeaf(leaf, query, k, 1.0f, null, parentBitSet);
                    TopDocs parallel = searchDiversifiedLeaf(leaf, query, k, 1.0f, newParallelContext(null), parentBitSet);
                    assertSameTopDocs(serial, parallel);
                    assertDistinctParents(parallel, parentBitSet);

                    float visitRatio = randomFloat() * 0.15f + 0.05f;
                    serial = searchDiversifiedLeaf(leaf, query, k, visitRatio, null, parentBitSet);
                    parallel = searchDiversifiedLeaf(leaf, query, k, visitRatio, newParallelContext(null), parentBitSet);
                    assertDistinctParents(parallel, parentBitSet);
                    // the parallel visit set is a superset, so it finds at least as many parents
                    assertTrue(parallel.scoreDocs.length >= serial.scoreDocs.length);
                    if (serial.scoreDocs.length == parallel.scoreDocs.length && serial.scoreDocs.length > 0) {
                        assertKthScoreDominance(serial, parallel);
                    }
                }
            }
        }
    }

    /**
     * Query-level sliced searches on a merge-sliced segment ({@code numSlices > 0}, only produced by merging): the
     * slice loop stays sequential but each slice's posting scan forks workers. The cancellation probe proves the
     * parallel path is actually reachable on such segments (it would pass vacuously if the codec gate declined).
     */
    public void testSlicedQueryOnMergeSlicedSegmentParity() throws IOException {
        int numDocs = randomIntBetween(3000, 5000);
        int dims = randomIntBetween(16, 48);
        int numSlices = randomIntBetween(2, 4);
        try (Directory dir = newDirectory()) {
            indexMergeSliced(dir, numDocs, dims, numSlices, randomBoolean());
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("merge-sliced coverage requires a single merged segment", 1, reader.leaves().size());
                IndexSearcher searcher = new IndexSearcher(reader);

                AbstractIVFKnnVectorQuery.IVFParallelismConfig cancelling = new AbstractIVFKnnVectorQuery.IVFParallelismConfig(
                    taskExecutor,
                    workers,
                    () -> {
                        throw new RuntimeException("cancelled");
                    }
                );
                Query probe = slicedQuery(randomVector(dims), 10, null, 1.0f, cancelling, allSliceRefs(numSlices));
                assertCancelled(expectThrows(RuntimeException.class, () -> searcher.search(probe, 10)));

                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    BytesRef[] sliceRefs = randomSliceRefs(numSlices);
                    Query filter = randomBoolean() ? new TermQuery(new Term(FILTER_FIELD, FILTER_MATCH)) : null;
                    TopDocs serial = searcher.search(slicedQuery(query, k, filter, 1.0f, null, sliceRefs), k);
                    TopDocs parallel = searcher.search(slicedQuery(query, k, filter, 1.0f, parallelismConfig(), sliceRefs), k);
                    assertSameTopDocs(serial, parallel);

                    float visitRatio = randomFloat() * 0.15f + 0.05f;
                    serial = searcher.search(slicedQuery(query, k, filter, visitRatio, null, sliceRefs), k);
                    parallel = searcher.search(slicedQuery(query, k, filter, visitRatio, parallelismConfig(), sliceRefs), k);
                    if (serial.scoreDocs.length == parallel.scoreDocs.length && serial.scoreDocs.length > 0) {
                        assertKthScoreDominance(serial, parallel);
                    }
                }
            }
        }
    }

    /** Diversified + sliced combined at query level on a merge-sliced segment. */
    public void testDiversifiedSlicedQueryParity() throws IOException {
        int numParents = randomIntBetween(400, 700);
        int dims = randomIntBetween(16, 48);
        int numSlices = randomIntBetween(2, 3);
        try (Directory dir = newDirectory()) {
            indexMergeSlicedWithParents(dir, numParents, dims, numSlices);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("merge-sliced coverage requires a single merged segment", 1, reader.leaves().size());
                IndexSearcher searcher = new IndexSearcher(reader);
                BitSetProducer parentsFilter = parentFilter(reader);
                BitSet parentBitSet = parentsFilter.getBitSet(getOnlyLeaf(reader));
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    BytesRef[] sliceRefs = randomSliceRefs(numSlices);
                    TopDocs serial = searcher.search(diversifiedSlicedQuery(query, k, parentsFilter, null, sliceRefs), k);
                    TopDocs parallel = searcher.search(diversifiedSlicedQuery(query, k, parentsFilter, parallelismConfig(), sliceRefs), k);
                    assertSameTopDocs(serial, parallel);
                    assertDistinctParents(parallel, parentBitSet);
                }
            }
        }
    }

    /**
     * A multi-segment diversified search (accumulators exist only when there is more than one leaf) must match the
     * accumulator-free per-leaf ground truth, both serially and through the full query-level parallel wiring
     * (diversified collector manager + per-worker diversifying collectors). This also guards the min-competitive
     * {@code heap.size() >= k()} fix: without the guards, an early under-filled-heap publish poisons the leaf's own
     * pruning floor via the fold-back in {@code nextVectorsBlock}, dropping any parent whose best child appears after
     * the heap fills with a score between the true k-th parent and the early published value. (The deterministic
     * demonstration of that mechanism is {@code IVFKnnSearchStrategyTests#testDiversifiedMaxScorePropagationRequiresFullHeap};
     * with natural data this end-to-end check rarely trips it because closest centroids are scanned first, so it
     * serves as the correctness net rather than the bug reproducer.)
     */
    public void testDiversifiedMultiSegmentDoesNotOverPrune() throws IOException {
        // Sized so the under-filled-heap publish actually happens: with 4-8 children per parent a 64-doc posting list
        // holds roughly 8-16 distinct parents, so a k above that leaves the heap under-filled at the first
        // once-per-posting-list publishes (where the unguarded code exported a bogus floor), while k below the
        // per-segment parent count guarantees the heap eventually fills and a poisoned floor would actually prune.
        int numParentsPerSegment = randomIntBetween(80, 120);
        int dims = randomIntBetween(16, 32);
        int k = randomIntBetween(25, 40);
        try (Directory dir = newDirectory()) {
            indexParentChildSegments(dir, numParentsPerSegment, dims, 2);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("the accumulator only exists with more than one leaf", 2, reader.leaves().size());
                IndexSearcher searcher = new IndexSearcher(reader);
                BitSetProducer parentsFilter = parentFilter(reader);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    Query knnQuery = new DiversifyingChildrenIVFKnnFloatVectorQuery(
                        FIELD,
                        query,
                        k,
                        k,
                        null,
                        parentsFilter,
                        1.0f,
                        testResolver()
                    ) {
                        @Override
                        IVFParallelismConfig intraSegmentParallelism() {
                            return null; // serial on purpose: the over-pruning bug is independent of parallelism
                        }
                    };
                    TopDocs actual = searcher.search(knnQuery, k);

                    Query parallelKnnQuery = new DiversifyingChildrenIVFKnnFloatVectorQuery(
                        FIELD,
                        query,
                        k,
                        k,
                        null,
                        parentsFilter,
                        1.0f,
                        testResolver()
                    ) {
                        @Override
                        IVFParallelismConfig intraSegmentParallelism() {
                            return parallelismConfig();
                        }
                    };
                    TopDocs actualParallel = searcher.search(parallelKnnQuery, k);

                    List<ScoreDoc> groundTruth = new ArrayList<>();
                    for (LeafReaderContext leaf : reader.leaves()) {
                        BitSet parentBitSet = parentsFilter.getBitSet(leaf);
                        // no parallel context and no accumulator: pruning uses only the leaf's own full-heap floor
                        TopDocs leafDocs = searchDiversifiedLeaf(leaf, query, k, 1.0f, null, parentBitSet);
                        for (ScoreDoc scoreDoc : leafDocs.scoreDocs) {
                            groundTruth.add(new ScoreDoc(scoreDoc.doc + leaf.docBase, scoreDoc.score));
                        }
                    }
                    groundTruth.sort(
                        Comparator.<ScoreDoc>comparingDouble(scoreDoc -> scoreDoc.score)
                            .reversed()
                            .thenComparingInt(scoreDoc -> scoreDoc.doc)
                    );
                    ScoreDoc[] expected = groundTruth.subList(0, Math.min(k, groundTruth.size())).toArray(ScoreDoc[]::new);
                    assertSameTopDocs(new TopDocs(actual.totalHits, expected), actual);
                    // the full query-level parallel wiring (diversified manager + worker factory) must agree too
                    assertSameTopDocs(new TopDocs(actualParallel.totalHits, expected), actualParallel);
                }
            }
        }
    }

    private IVFParallelScanContext newParallelContext(Runnable checkCancelled) {
        return new IVFParallelScanContext(
            taskExecutor,
            workers,
            checkCancelled == null ? () -> {} : checkCancelled,
            new LongAccumulator(Long::max, AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE)
        );
    }

    /**
     * Runs one leaf-level IVF search, mirroring how {@link IVFKnnFloatVectorQuery#approximateSearch} drives the codec
     * reader, with an optional parallel scan context and an optional pre-filter bit set.
     */
    private TopDocs searchLeaf(
        LeafReaderContext leaf,
        float[] query,
        int k,
        float visitRatio,
        IVFParallelScanContext parallelScanContext,
        FixedBitSet filter
    ) throws IOException {
        // sometimes mirror the multi-leaf production wiring, where the leaf strategy shares the workers' accumulator
        LongAccumulator leafAccumulator = parallelScanContext != null && randomBoolean() ? parallelScanContext.workerAccumulator() : null;
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, k, k, leafAccumulator, parallelScanContext);
        MaxScoreTopKnnCollector collector = new MaxScoreTopKnnCollector(k, Integer.MAX_VALUE, strategy);
        strategy.setCollector(collector);
        Bits liveDocs = leaf.reader().getLiveDocs();
        int maxDoc = leaf.reader().maxDoc();
        ESAcceptDocs acceptDocs;
        if (filter != null) {
            FixedBitSet filtered = filter.clone();
            if (liveDocs != null) {
                liveDocs.applyMask(filtered, 0);
            }
            acceptDocs = new ESAcceptDocs.BitsAcceptDocs(filtered, maxDoc);
        } else if (liveDocs != null) {
            acceptDocs = new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc);
        } else {
            acceptDocs = new ESAcceptDocs.ESAcceptDocsAll();
        }
        leaf.reader().searchNearestVectors(FIELD, query, collector, acceptDocs);
        return collector.topDocs();
    }

    /**
     * Indexes {@code numVectors} vectors into a single segment using the ESNext DiskBBQ format with small clusters,
     * so a search drains enough ranked posting lists to actually fork workers. Optionally deletes some documents
     * (keeping the single segment intact via {@link NoMergePolicy}).
     */
    private void indexSingleSegment(Directory dir, int numVectors, int dims, int deletes) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(
            TestUtil.alwaysKnnVectorsFormat(new ESNextDiskBBQVectorsFormat(64, 16, null))
        )
            .setMaxBufferedDocs(numVectors * 2)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int i = 0; i < numVectors; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField("id", "id" + i, Field.Store.NO));
                writer.addDocument(doc);
            }
            writer.commit();
            for (int i = 0; i < deletes; i++) {
                writer.deleteDocuments(new Term("id", "id" + randomInt(numVectors - 1)));
            }
            writer.commit();
        }
    }

    /**
     * Runs one leaf-level diversified (parent/child) IVF search, mirroring how the diversified collector manager
     * wires the leaf collector, with an optional parallel scan context.
     */
    private TopDocs searchDiversifiedLeaf(
        LeafReaderContext leaf,
        float[] query,
        int k,
        float visitRatio,
        IVFParallelScanContext parallelScanContext,
        BitSet parentBitSet
    ) throws IOException {
        LongAccumulator leafAccumulator = parallelScanContext != null && randomBoolean() ? parallelScanContext.workerAccumulator() : null;
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, k, k, leafAccumulator, parallelScanContext);
        DiversifyingNearestChildrenKnnCollector collector = new DiversifyingNearestChildrenKnnCollector(
            k,
            Integer.MAX_VALUE,
            strategy,
            parentBitSet
        );
        strategy.setCollector(collector);
        leaf.reader().searchNearestVectors(FIELD, query, collector, new ESAcceptDocs.ESAcceptDocsAll());
        return collector.topDocs();
    }

    private static BitSetProducer parentFilter(IndexReader reader) throws IOException {
        BitSetProducer parentsFilter = new QueryBitSetProducer(new TermQuery(new Term("docType", "_parent")));
        CheckJoinIndex.check(reader, parentsFilter);
        return parentsFilter;
    }

    /**
     * Indexes {@code numSegments} segments of parent/child blocks (children carry the vectors, the trailing parent
     * doc carries only the marker field), kept separate via {@link NoMergePolicy}.
     */
    private void indexParentChildSegments(Directory dir, int parentsPerSegment, int dims, int numSegments) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(
            TestUtil.alwaysKnnVectorsFormat(new ESNextDiskBBQVectorsFormat(64, 16, null))
        )
            .setMaxBufferedDocs(parentsPerSegment * 20)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int segment = 0; segment < numSegments; segment++) {
                for (int parent = 0; parent < parentsPerSegment; parent++) {
                    List<Document> block = new ArrayList<>();
                    int children = randomIntBetween(4, 8);
                    for (int child = 0; child < children; child++) {
                        Document childDoc = new Document();
                        childDoc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                        block.add(childDoc);
                    }
                    Document parentDoc = new Document();
                    parentDoc.add(new StringField("docType", "_parent", Field.Store.NO));
                    block.add(parentDoc);
                    writer.addDocuments(block);
                }
                writer.commit();
            }
        }
    }

    /**
     * Indexes a merge-sliced segment: two flushed segments force-merged into one, which is the only path on which the
     * writer records {@code numSlices > 0} (a flush with a slice field writes {@code numSlices == 0}).
     */
    private void indexMergeSliced(Directory dir, int numDocs, int dims, int numSlices, boolean deletes) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(
            TestUtil.alwaysKnnVectorsFormat(new ESNextDiskBBQVectorsFormat(64, randomIntBetween(4, 16), SLICE_FIELD))
        ).setIndexSort(new Sort(new SortField(SLICE_FIELD, SortField.Type.STRING)));
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int i = 0; i < numDocs; i++) {
                Document doc = new Document();
                doc.add(SortedDocValuesField.indexedField(SLICE_FIELD, new BytesRef("" + randomInt(numSlices - 1))));
                doc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(FILTER_FIELD, randomBoolean() ? FILTER_MATCH : "miss", Field.Store.NO));
                doc.add(new StringField("id", "id" + i, Field.Store.NO));
                writer.addDocument(doc);
                if (i == numDocs / 2) {
                    writer.commit();
                }
            }
            writer.commit();
            writer.forceMerge(1);
            if (deletes) {
                for (int i = 0; i < numDocs / 10; i++) {
                    writer.deleteDocuments(new Term("id", "id" + randomInt(numDocs - 1)));
                }
            }
            writer.commit();
        }
    }

    /** Like {@link #indexMergeSliced} but with parent/child blocks, every doc of a block sharing the same slice. */
    private void indexMergeSlicedWithParents(Directory dir, int numParents, int dims, int numSlices) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(
            TestUtil.alwaysKnnVectorsFormat(new ESNextDiskBBQVectorsFormat(64, randomIntBetween(4, 16), SLICE_FIELD))
        ).setIndexSort(new Sort(new SortField(SLICE_FIELD, SortField.Type.STRING))).setParentField(Engine.ROOT_DOC_FIELD_NAME);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int parent = 0; parent < numParents; parent++) {
                BytesRef slice = new BytesRef("" + randomInt(numSlices - 1));
                List<Document> block = new ArrayList<>();
                int children = randomIntBetween(1, 5);
                for (int child = 0; child < children; child++) {
                    Document childDoc = new Document();
                    childDoc.add(SortedDocValuesField.indexedField(SLICE_FIELD, slice));
                    childDoc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                    block.add(childDoc);
                }
                Document parentDoc = new Document();
                parentDoc.add(SortedDocValuesField.indexedField(SLICE_FIELD, slice));
                parentDoc.add(new StringField("docType", "_parent", Field.Store.NO));
                block.add(parentDoc);
                writer.addDocuments(block);
                if (parent == numParents / 2) {
                    writer.commit();
                }
            }
            writer.commit();
            writer.forceMerge(1);
        }
    }

    private TestIvfQueryConfigResolver testResolver() {
        return new TestIvfQueryConfigResolver(
            ESNextDiskBBQVectorsFormat.CentroidIndexFormat.FLAT,
            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
            false,
            1.0f
        );
    }

    private AbstractIVFKnnVectorQuery.IVFParallelismConfig parallelismConfig() {
        return new AbstractIVFKnnVectorQuery.IVFParallelismConfig(taskExecutor, workers, () -> {});
    }

    /**
     * Builds a sliced query whose parallelism is controlled directly by the test ({@code null} forces the serial
     * path) instead of through the {@code es.vectors.*} system properties, which are static finals read at class
     * load and therefore unreliable inside a shared test JVM.
     */
    private IVFKnnFloatSlicedVectorQuery slicedQuery(
        float[] vector,
        int k,
        Query filter,
        float visitRatio,
        AbstractIVFKnnVectorQuery.IVFParallelismConfig parallelismConfig,
        BytesRef... sliceIds
    ) {
        return new IVFKnnFloatSlicedVectorQuery(FIELD, vector, k, k, filter, visitRatio, testResolver(), SLICE_FIELD, sliceIds) {
            @Override
            IVFParallelismConfig intraSegmentParallelism() {
                return parallelismConfig;
            }
        };
    }

    /** See {@link #slicedQuery}. */
    private DiversifyingChildrenIVFKnnFloatSlicedVectorQuery diversifiedSlicedQuery(
        float[] vector,
        int k,
        BitSetProducer parentsFilter,
        AbstractIVFKnnVectorQuery.IVFParallelismConfig parallelismConfig,
        BytesRef... sliceIds
    ) {
        return new DiversifyingChildrenIVFKnnFloatSlicedVectorQuery(
            FIELD,
            vector,
            k,
            k,
            null,
            parentsFilter,
            1.0f,
            testResolver(),
            SLICE_FIELD,
            sliceIds
        ) {
            @Override
            IVFParallelismConfig intraSegmentParallelism() {
                return parallelismConfig;
            }
        };
    }

    private static BytesRef[] allSliceRefs(int numSlices) {
        BytesRef[] refs = new BytesRef[numSlices];
        for (int i = 0; i < numSlices; i++) {
            refs[i] = new BytesRef("" + i);
        }
        return refs;
    }

    /** An empty array means "search all slices" per the sliced query's contract. */
    private static BytesRef[] randomSliceRefs(int numSlices) {
        if (randomBoolean()) {
            return new BytesRef[0];
        }
        Set<Integer> picked = new HashSet<>();
        int count = randomIntBetween(1, numSlices);
        while (picked.size() < count) {
            picked.add(randomInt(numSlices - 1));
        }
        return picked.stream().map(slice -> new BytesRef("" + slice)).toArray(BytesRef[]::new);
    }

    private static void assertCancelled(RuntimeException e) {
        for (Throwable t = e; t != null; t = t.getCause() == t ? null : t.getCause()) {
            if (t.getMessage() != null && t.getMessage().contains("cancelled")) {
                return;
            }
        }
        throw new AssertionError("expected a cancellation failure", e);
    }

    /** With a partial visit budget the parallel visit set is a superset of the serial one, so its k-th best score dominates. */
    private static void assertKthScoreDominance(TopDocs serial, TopDocs parallel) {
        assertEquals(serial.scoreDocs.length, parallel.scoreDocs.length);
        if (serial.scoreDocs.length == 0) {
            return;
        }
        float serialMin = serial.scoreDocs[serial.scoreDocs.length - 1].score;
        float parallelMin = parallel.scoreDocs[parallel.scoreDocs.length - 1].score;
        assertTrue("parallel k-th score " + parallelMin + " must be >= serial k-th score " + serialMin, parallelMin >= serialMin);
    }

    private static void assertDistinctParents(TopDocs topDocs, BitSet parentBitSet) {
        Set<Integer> parents = new HashSet<>();
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            assertFalse("a returned doc must be a child", parentBitSet.get(scoreDoc.doc));
            assertTrue("each returned child must map to a distinct parent", parents.add(parentBitSet.nextSetBit(scoreDoc.doc)));
        }
    }

    /**
     * Asserts the two results agree rank-for-rank on scores, and on docs up to permutations within runs of exactly
     * tied scores. Block-level pruning compares scores only while queue ordering tiebreaks on doc id, so the parallel
     * run's floor-propagation timing may legitimately reorder (or, at the k boundary, swap) exactly tied docs; the
     * final tied run is therefore compared by score only.
     */
    private void assertSameTopDocs(TopDocs serial, TopDocs parallel) {
        assertEquals(serial.scoreDocs.length, parallel.scoreDocs.length);
        int count = serial.scoreDocs.length;
        for (int d = 0; d < count; d++) {
            assertEquals("score at rank " + d, serial.scoreDocs[d].score, parallel.scoreDocs[d].score, 0.0f);
        }
        int start = 0;
        while (start < count) {
            int end = start + 1;
            while (end < count && serial.scoreDocs[end].score == serial.scoreDocs[start].score) {
                end++;
            }
            if (end < count) { // the run touching the k boundary may differ in tied docs, not just their order
                Set<Integer> serialDocs = new HashSet<>();
                Set<Integer> parallelDocs = new HashSet<>();
                for (int d = start; d < end; d++) {
                    serialDocs.add(serial.scoreDocs[d].doc);
                    parallelDocs.add(parallel.scoreDocs[d].doc);
                }
                assertEquals("docs in tied-score run [" + start + "," + end + ")", serialDocs, parallelDocs);
            }
            start = end;
        }
    }

    private LeafReaderContext getOnlyLeaf(DirectoryReader reader) {
        assertEquals("test expects a single segment", 1, reader.leaves().size());
        return reader.leaves().get(0);
    }

    private float[] randomVector(int dims) {
        float[] vector = new float[dims];
        for (int i = 0; i < dims; i++) {
            vector[i] = randomFloat() * 2 - 1;
        }
        return vector;
    }
}
