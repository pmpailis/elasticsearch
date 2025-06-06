/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.snapshots.mockstore;

import com.carrotsearch.randomizedtesting.RandomizedContext;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.CorruptIndexException;
import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.cluster.metadata.ProjectId;
import org.elasticsearch.cluster.metadata.RepositoryMetadata;
import org.elasticsearch.cluster.service.ClusterService;
import org.elasticsearch.common.blobstore.BlobContainer;
import org.elasticsearch.common.blobstore.BlobPath;
import org.elasticsearch.common.blobstore.BlobStore;
import org.elasticsearch.common.blobstore.DeleteResult;
import org.elasticsearch.common.blobstore.OperationPurpose;
import org.elasticsearch.common.blobstore.fs.FsBlobContainer;
import org.elasticsearch.common.blobstore.support.BlobMetadata;
import org.elasticsearch.common.blobstore.support.FilterBlobContainer;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.collect.Iterators;
import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Setting.Property;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.BigArrays;
import org.elasticsearch.common.util.set.Sets;
import org.elasticsearch.core.CheckedConsumer;
import org.elasticsearch.core.PathUtils;
import org.elasticsearch.env.Environment;
import org.elasticsearch.indices.recovery.RecoverySettings;
import org.elasticsearch.plugins.RepositoryPlugin;
import org.elasticsearch.repositories.RepositoriesMetrics;
import org.elasticsearch.repositories.Repository;
import org.elasticsearch.repositories.blobstore.BlobStoreRepository;
import org.elasticsearch.repositories.fs.FsRepository;
import org.elasticsearch.xcontent.NamedXContentRegistry;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.regex.Pattern;

public class MockRepository extends FsRepository {
    private static final Logger logger = LogManager.getLogger(MockRepository.class);

    public static class Plugin extends org.elasticsearch.plugins.Plugin implements RepositoryPlugin {

        public static final Setting<String> USERNAME_SETTING = Setting.simpleString("secret.mock.username", Property.NodeScope);
        public static final Setting<String> PASSWORD_SETTING = Setting.simpleString(
            "secret.mock.password",
            Property.NodeScope,
            Property.Filtered
        );

        @Override
        public Map<String, Repository.Factory> getRepositories(
            Environment env,
            NamedXContentRegistry namedXContentRegistry,
            ClusterService clusterService,
            BigArrays bigArrays,
            RecoverySettings recoverySettings,
            RepositoriesMetrics repositoriesMetrics
        ) {
            return Collections.singletonMap(
                "mock",
                (projectId, metadata) -> new MockRepository(
                    projectId,
                    metadata,
                    env,
                    namedXContentRegistry,
                    clusterService,
                    bigArrays,
                    recoverySettings
                )
            );
        }

        @Override
        public List<Setting<?>> getSettings() {
            return Arrays.asList(USERNAME_SETTING, PASSWORD_SETTING);
        }
    }

    /**
     * Setting name for a setting that can be updated dynamically to test {@link #canUpdateInPlace(Settings, Set)}.
     */
    public static final String DUMMY_UPDATABLE_SETTING_NAME = "dummy_setting";

    private final AtomicLong failureCounter = new AtomicLong();

    public long getFailureCount() {
        return failureCounter.get();
    }

    private volatile double randomControlIOExceptionRate;

    private volatile double randomDataFileIOExceptionRate;
    private volatile Predicate<String> randomIOExceptionPattern;

    private volatile boolean useLuceneCorruptionException;

    private volatile long maximumNumberOfFailures;

    private final long waitAfterUnblock;

    private final String randomPrefix;

    private final Environment env;

    private volatile boolean blockOnAnyFiles;

    private volatile boolean blockOnDataFiles;

    private volatile boolean blockAndFailOnDataFiles;

    private volatile boolean blockOnDeleteIndexN;

    /**
     * Allows blocking on writing the index-N blob and subsequently failing it on unblock.
     * This is a way to enforce blocking the finalization of a snapshot, while permitting other IO operations to proceed unblocked.
     */
    private volatile boolean blockAndFailOnWriteIndexFile;

    /**
     * Same as {@link #blockAndFailOnWriteIndexFile} but proceeds without error after unblock.
     */
    private volatile boolean blockOnWriteIndexFile;

    /** Allows blocking on writing the snapshot file at the end of snapshot creation to simulate a died master node */
    private volatile boolean blockAndFailOnWriteSnapFile;
    private volatile String blockedIndexId;

    private volatile boolean blockOnWriteShardLevelMeta;

    private volatile boolean blockAndFailOnWriteShardLevelMeta;

    private volatile boolean blockOnReadIndexMeta;

    private final AtomicBoolean blockOnceOnReadSnapshotInfo = new AtomicBoolean(false);

    /**
     * Writes to the blob {@code index.latest} at the repository root will fail with an {@link IOException} if {@code true}.
     */
    private volatile boolean failOnIndexLatest = false;

    /**
     * Reading blobs will fail with an {@link AssertionError} once the repository has been blocked once.
     */
    private volatile boolean failReadsAfterUnblock;
    private volatile boolean throwReadErrorAfterUnblock = false;

    private volatile boolean blockAndFailOnReadSnapFile;

    private volatile boolean blockAndFailOnReadIndexFile;

    private volatile boolean blocked = false;

    private volatile boolean failOnDeleteContainer = false;

    public MockRepository(
        ProjectId projectId,
        RepositoryMetadata metadata,
        Environment environment,
        NamedXContentRegistry namedXContentRegistry,
        ClusterService clusterService,
        BigArrays bigArrays,
        RecoverySettings recoverySettings
    ) {
        super(
            projectId,
            overrideSettings(metadata, environment),
            environment,
            namedXContentRegistry,
            clusterService,
            bigArrays,
            recoverySettings
        );
        randomControlIOExceptionRate = metadata.settings().getAsDouble("random_control_io_exception_rate", 0.0);
        randomDataFileIOExceptionRate = metadata.settings().getAsDouble("random_data_file_io_exception_rate", 0.0);
        randomIOExceptionPattern = Pattern.compile(metadata.settings().get("random_io_exception_pattern", ".*")).asMatchPredicate();
        useLuceneCorruptionException = metadata.settings().getAsBoolean("use_lucene_corruption", false);
        maximumNumberOfFailures = metadata.settings().getAsLong("max_failure_number", 100L);
        blockOnAnyFiles = metadata.settings().getAsBoolean("block_on_control", false);
        blockOnDataFiles = metadata.settings().getAsBoolean("block_on_data", false);
        blockAndFailOnWriteSnapFile = metadata.settings().getAsBoolean("block_on_snap", false);
        randomPrefix = metadata.settings().get("random", "default");
        waitAfterUnblock = metadata.settings().getAsLong("wait_after_unblock", 0L);
        env = environment;
        logger.info("starting mock repository with random prefix {}", randomPrefix);
    }

    @Override
    public RepositoryMetadata getMetadata() {
        return overrideSettings(super.getMetadata(), env);
    }

    @Override
    public boolean canUpdateInPlace(Settings updatedSettings, Set<String> ignoredSettings) {
        // allow updating dummy setting for test purposes
        return super.canUpdateInPlace(updatedSettings, Sets.union(ignoredSettings, Set.of(DUMMY_UPDATABLE_SETTING_NAME)));
    }

    private static RepositoryMetadata overrideSettings(RepositoryMetadata metadata, Environment environment) {
        // TODO: use another method of testing not being able to read the test file written by the master...
        // this is super duper hacky
        if (metadata.settings().getAsBoolean("localize_location", false)) {
            Path location = PathUtils.get(metadata.settings().get("location"));
            location = location.resolve(Integer.toString(environment.hashCode()));
            return new RepositoryMetadata(
                metadata.name(),
                metadata.type(),
                Settings.builder().put(metadata.settings()).put("location", location.toAbsolutePath()).build()
            );
        } else {
            return metadata;
        }
    }

    private long incrementAndGetFailureCount() {
        return failureCounter.incrementAndGet();
    }

    @Override
    protected void doStop() {
        unblock();
        super.doStop();
    }

    @Override
    protected BlobStore createBlobStore() throws Exception {
        return new MockBlobStore(super.createBlobStore());
    }

    public synchronized void unblock() {
        blocked = false;
        // Clean blocking flags, so we wouldn't try to block again
        blockOnDataFiles = false;
        blockAndFailOnDataFiles = false;
        blockOnAnyFiles = false;
        blockAndFailOnWriteIndexFile = false;
        blockOnWriteIndexFile = false;
        blockAndFailOnWriteSnapFile = false;
        blockedIndexId = null;
        blockOnDeleteIndexN = false;
        blockOnWriteShardLevelMeta = false;
        blockAndFailOnWriteShardLevelMeta = false;
        blockOnReadIndexMeta = false;
        blockOnceOnReadSnapshotInfo.set(false);
        blockAndFailOnReadSnapFile = false;
        blockAndFailOnReadIndexFile = false;
        this.notifyAll();
    }

    public void setMaximumNumberOfFailures(long maximumNumberOfFailures) {
        logger.debug("Setting maximum number of failures to [{}]", maximumNumberOfFailures);
        this.maximumNumberOfFailures = maximumNumberOfFailures;
    }

    public void setUseLuceneCorruptionException(boolean useLuceneCorruptionException) {
        logger.debug("Setting using lucene corruption exception to [{}]", useLuceneCorruptionException);
        this.useLuceneCorruptionException = useLuceneCorruptionException;
    }

    public void setRandomControlIOExceptionRate(double randomControlIOExceptionRate) {
        logger.debug("Setting random control I/O exception rate to [{}]", randomControlIOExceptionRate);
        this.randomControlIOExceptionRate = randomControlIOExceptionRate;
    }

    public void setRandomDataFileIOExceptionRate(double randomDataFileIOExceptionRate) {
        logger.debug("Setting random data file I/O exception rate to [{}]", randomDataFileIOExceptionRate);
        this.randomDataFileIOExceptionRate = randomDataFileIOExceptionRate;
    }

    public void setRandomIOExceptionPattern(String randomIOExceptionPattern) {
        logger.debug("Setting random I/O exception pattern to [{}]", randomIOExceptionPattern);
        this.randomIOExceptionPattern = Pattern.compile(randomIOExceptionPattern).asMatchPredicate();
    }

    public void blockOnDataFiles() {
        assert blockAndFailOnDataFiles == false : "Either fail or wait after data file, not both";
        blockOnDataFiles = true;
    }

    public void blockAndFailOnDataFiles() {
        assert blockOnDataFiles == false : "Either fail or wait after data file, not both";
        blockAndFailOnDataFiles = true;
    }

    public void setBlockOnAnyFiles() {
        blockOnAnyFiles = true;
    }

    public void setBlockAndFailOnWriteSnapFiles() {
        blockAndFailOnWriteSnapFile = true;
    }

    public void setBlockOnShardLevelSnapFiles(String indexId) {
        blockedIndexId = indexId;
    }

    public void setBlockAndFailOnWriteIndexFile() {
        assert blockOnWriteIndexFile == false : "Either fail or wait after blocking on index-N not both";
        blockAndFailOnWriteIndexFile = true;
    }

    public void setBlockOnWriteIndexFile() {
        assert blockAndFailOnWriteIndexFile == false : "Either fail or wait after blocking on index-N not both";
        blockOnWriteIndexFile = true;
    }

    public void setBlockOnDeleteIndexFile() {
        blockOnDeleteIndexN = true;
    }

    public void setBlockOnWriteShardLevelMeta() {
        assert blockAndFailOnWriteShardLevelMeta == false : "Either fail or wait after blocking on shard level metadata, not both";
        blockOnWriteShardLevelMeta = true;
    }

    public void setBlockAndFailOnWriteShardLevelMeta() {
        assert blockOnWriteShardLevelMeta == false : "Either fail or wait after blocking on shard level metadata, not both";
        blockAndFailOnWriteShardLevelMeta = true;
    }

    public void setBlockOnReadIndexMeta() {
        blockOnReadIndexMeta = true;
    }

    public void setFailReadsAfterUnblock(boolean failReadsAfterUnblock) {
        this.failReadsAfterUnblock = failReadsAfterUnblock;
    }

    public void setBlockAndFailOnReadSnapFiles() {
        blockAndFailOnReadSnapFile = true;
    }

    public void setBlockAndFailOnReadIndexFiles() {
        blockAndFailOnReadIndexFile = true;
    }

    /**
     * Enable blocking a single read of {@link org.elasticsearch.snapshots.SnapshotInfo} in case the repo is already blocked on another
     * file. This allows testing very specific timing issues where a read of {@code SnapshotInfo} is much slower than another concurrent
     * repository operation. See {@link #blockExecution()} for the exact mechanics of why we need a secondary block defined here.
     * TODO: clean this up to not require a second block set
     */
    public void setBlockOnceOnReadSnapshotInfoIfAlreadyBlocked() {
        blockOnceOnReadSnapshotInfo.set(true);
    }

    /**
     * Sets the fail-on-delete-container flag, which if {@code true} throws an exception when deleting a {@link BlobContainer}.
     */
    public void setFailOnDeleteContainer(boolean failOnDeleteContainer) {
        this.failOnDeleteContainer = failOnDeleteContainer;
    }

    public boolean blocked() {
        return blocked;
    }

    public void setFailOnIndexLatest(boolean failOnIndexLatest) {
        this.failOnIndexLatest = failOnIndexLatest;
    }

    private synchronized boolean blockExecution() {
        logger.debug("[{}] Blocking execution", metadata.name());
        boolean wasBlocked = false;
        try {
            while (blockAndFailOnDataFiles
                || blockOnDataFiles
                || blockOnAnyFiles
                || blockAndFailOnWriteIndexFile
                || blockOnWriteIndexFile
                || blockAndFailOnWriteSnapFile
                || blockOnDeleteIndexN
                || blockOnWriteShardLevelMeta
                || blockAndFailOnWriteShardLevelMeta
                || blockOnReadIndexMeta
                || blockAndFailOnReadSnapFile
                || blockAndFailOnReadIndexFile
                || blockedIndexId != null) {
                blocked = true;
                this.wait();
                wasBlocked = true;
            }
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
        logger.debug("[{}] Unblocking execution", metadata.name());
        if (wasBlocked && failReadsAfterUnblock) {
            logger.debug("[{}] Next read operations will fail", metadata.name());
            this.throwReadErrorAfterUnblock = true;
        }
        return wasBlocked;
    }

    public class MockBlobStore extends BlobStoreWrapper {
        ConcurrentMap<String, AtomicLong> accessCounts = new ConcurrentHashMap<>();

        private long incrementAndGet(String path) {
            AtomicLong value = accessCounts.get(path);
            if (value == null) {
                value = accessCounts.putIfAbsent(path, new AtomicLong(1));
            }
            if (value != null) {
                return value.incrementAndGet();
            }
            return 1;
        }

        public MockBlobStore(BlobStore delegate) {
            super(delegate);
        }

        @Override
        public BlobContainer blobContainer(BlobPath path) {
            return new MockBlobContainer(super.blobContainer(path));
        }

        private class MockBlobContainer extends FilterBlobContainer {

            private boolean shouldFail(String blobName, double probability) {
                if (probability > 0.0) {
                    String path = path().add(blobName).buildAsString() + randomPrefix;
                    path += "/" + incrementAndGet(path);
                    if (Math.abs(hashCode(path)) < Integer.MAX_VALUE * probability) {
                        if (randomIOExceptionPattern.test(path)) {
                            if (incrementAndGetFailureCount() <= maximumNumberOfFailures) {
                                logger.info("failing [{}]", path);
                                return true;
                            } else {
                                logger.info("did not fail [{}] because failure count larger than maximum", path);
                            }
                        } else {
                            logger.info("did not fail [{}] because it does not match the pattern", path);
                        }
                    } else {
                        logger.info("did not fail [{}] due to probability", path);
                    }
                }
                return false;
            }

            private int hashCode(String path) {
                try {
                    MessageDigest digest = MessageDigest.getInstance("MD5");
                    byte[] bytes = digest.digest(path.getBytes(StandardCharsets.UTF_8));
                    int i = 0;
                    return ((bytes[i++] & 0xFF) << 24) | ((bytes[i++] & 0xFF) << 16) | ((bytes[i++] & 0xFF) << 8) | (bytes[i++] & 0xFF);
                } catch (NoSuchAlgorithmException ex) {
                    throw new ElasticsearchException("cannot calculate hashcode", ex);
                }
            }

            private void maybeIOExceptionOrBlock(String blobName) throws IOException {
                if (INDEX_LATEST_BLOB.equals(blobName)) {
                    // Don't mess with the index.latest blob here, failures to write to it are ignored by upstream logic and we have
                    // specific tests that cover the error handling around this blob.
                    return;
                }
                if (blobName.startsWith("__")) {
                    if (shouldFail(blobName, randomDataFileIOExceptionRate)) {
                        logger.info("throwing random IOException for file [{}] at path [{}]", blobName, path());
                        if (useLuceneCorruptionException) {
                            throw new CorruptIndexException("Random corruption", "random file");
                        } else {
                            throw new IOException("Random IOException");
                        }
                    } else if (blockOnDataFiles) {
                        blockExecutionAndMaybeWait(blobName);
                    } else if (blockAndFailOnDataFiles) {
                        blockExecutionAndFail(blobName);
                    }
                } else {
                    if (shouldFail(blobName, randomControlIOExceptionRate)) {
                        logger.info("throwing random IOException for file [{}] at path [{}]", blobName, path());
                        throw new IOException("Random IOException");
                    } else if (blockOnAnyFiles) {
                        blockExecutionAndMaybeWait(blobName);
                    } else if (blobName.startsWith(SNAPSHOT_PREFIX) && (blockAndFailOnWriteSnapFile || blockAndFailOnReadSnapFile)) {
                        blockExecutionAndFail(blobName);
                    } else if (blobName.startsWith(INDEX_FILE_PREFIX) && blockAndFailOnReadIndexFile) {
                        blockExecutionAndFail(blobName);
                    } else if (blockedIndexId != null && path().parts().contains(blockedIndexId) && blobName.startsWith(SNAPSHOT_PREFIX)) {
                        blockExecutionAndMaybeWait(blobName);
                    }
                }
            }

            private void blockExecutionAndMaybeWait(final String blobName) throws IOException {
                logger.info("[{}] blocking I/O operation for file [{}] at path [{}]", metadata.name(), blobName, path());
                final boolean wasBlocked = blockExecution();
                if (wasBlocked && lifecycle.stoppedOrClosed()) {
                    throw new IOException("already closed");
                }
                if (wasBlocked && waitAfterUnblock > 0) {
                    try {
                        // Delay operation after unblocking
                        // So, we can start node shutdown while this operation is still running.
                        Thread.sleep(waitAfterUnblock);
                    } catch (InterruptedException ex) {
                        //
                    }
                }
            }

            /**
             * Blocks an I/O operation on the blob fails and throws an exception when unblocked
             */
            private void blockExecutionAndFail(final String blobName) throws IOException {
                logger.info("blocking I/O operation for file [{}] at path [{}]", blobName, path());
                blockExecution();
                throw new IOException("exception after block");
            }

            private void maybeReadErrorAfterBlock(final String blobName) {
                if (throwReadErrorAfterUnblock) {
                    throw new AssertionError("Read operation are not allowed anymore at this point [blob=" + blobName + "]");
                }
            }

            MockBlobContainer(BlobContainer delegate) {
                super(delegate);
            }

            @Override
            protected BlobContainer wrapChild(BlobContainer child) {
                return new MockBlobContainer(child);
            }

            @Override
            public InputStream readBlob(OperationPurpose purpose, String name) throws IOException {
                if (blockOnReadIndexMeta && name.startsWith(BlobStoreRepository.METADATA_PREFIX) && path().equals(basePath()) == false) {
                    blockExecutionAndMaybeWait(name);
                } else if (path().equals(basePath())
                    && name.startsWith(BlobStoreRepository.SNAPSHOT_PREFIX)
                    && blockOnceOnReadSnapshotInfo.compareAndSet(true, false)) {
                        blockExecutionAndMaybeWait(name);
                    } else {
                        maybeReadErrorAfterBlock(name);
                        maybeIOExceptionOrBlock(name);
                    }
                return super.readBlob(purpose, name);
            }

            @Override
            public InputStream readBlob(OperationPurpose purpose, String name, long position, long length) throws IOException {
                maybeReadErrorAfterBlock(name);
                maybeIOExceptionOrBlock(name);
                return super.readBlob(purpose, name, position, length);
            }

            @Override
            public DeleteResult delete(OperationPurpose purpose) throws IOException {
                if (failOnDeleteContainer) {
                    throw new IOException("simulated delete-container failure");
                }
                DeleteResult deleteResult = DeleteResult.ZERO;
                for (BlobContainer child : children(purpose).values()) {
                    deleteResult = deleteResult.add(child.delete(purpose));
                }
                final Map<String, BlobMetadata> blobs = listBlobs(purpose);
                long deleteBlobCount = blobs.size();
                long deleteByteCount = 0L;
                for (String blob : blobs.values().stream().map(BlobMetadata::name).toList()) {
                    maybeIOExceptionOrBlock(blob);
                    deleteBlobsIgnoringIfNotExists(purpose, Iterators.single(blob));
                    deleteByteCount += blobs.get(blob).length();
                }
                blobStore().blobContainer(path().parent())
                    .deleteBlobsIgnoringIfNotExists(purpose, Iterators.single(path().parts().get(path().parts().size() - 1)));
                return deleteResult.add(deleteBlobCount, deleteByteCount);
            }

            @Override
            public void deleteBlobsIgnoringIfNotExists(OperationPurpose purpose, Iterator<String> blobNames) throws IOException {
                final List<String> names = new ArrayList<>();
                blobNames.forEachRemaining(names::add);
                if (blockOnDeleteIndexN && names.stream().anyMatch(name -> name.startsWith(BlobStoreRepository.INDEX_FILE_PREFIX))) {
                    blockExecutionAndMaybeWait("index-{N}");
                }
                super.deleteBlobsIgnoringIfNotExists(purpose, names.iterator());
            }

            @Override
            public Map<String, BlobMetadata> listBlobs(OperationPurpose purpose) throws IOException {
                maybeIOExceptionOrBlock("");
                return super.listBlobs(purpose);
            }

            @Override
            public Map<String, BlobContainer> children(OperationPurpose purpose) throws IOException {
                final Map<String, BlobContainer> res = new HashMap<>();
                for (Map.Entry<String, BlobContainer> entry : super.children(purpose).entrySet()) {
                    res.put(entry.getKey(), new MockBlobContainer(entry.getValue()));
                }
                return res;
            }

            @Override
            public Map<String, BlobMetadata> listBlobsByPrefix(OperationPurpose purpose, String blobNamePrefix) throws IOException {
                maybeIOExceptionOrBlock(blobNamePrefix);
                return super.listBlobsByPrefix(purpose, blobNamePrefix);
            }

            @Override
            public void writeBlob(
                OperationPurpose purpose,
                String blobName,
                InputStream inputStream,
                long blobSize,
                boolean failIfAlreadyExists
            ) throws IOException {
                beforeWrite(blobName);
                super.writeBlob(purpose, blobName, inputStream, blobSize, failIfAlreadyExists);
                if (RandomizedContext.current().getRandom().nextBoolean()) {
                    // for network based repositories, the blob may have been written but we may still
                    // get an error with the client connection, so an IOException here simulates this
                    maybeIOExceptionOrBlock(blobName);
                }
            }

            @Override
            public void writeMetadataBlob(
                OperationPurpose purpose,
                String blobName,
                boolean failIfAlreadyExists,
                boolean atomic,
                CheckedConsumer<OutputStream, IOException> writer
            ) throws IOException {
                if (atomic) {
                    beforeAtomicWrite(blobName);
                } else {
                    beforeWrite(blobName);
                }
                super.writeMetadataBlob(purpose, blobName, failIfAlreadyExists, atomic, writer);
                if (RandomizedContext.current().getRandom().nextBoolean()) {
                    // for network based repositories, the blob may have been written but we may still
                    // get an error with the client connection, so an IOException here simulates this
                    maybeIOExceptionOrBlock(blobName);
                }
            }

            private void beforeWrite(String blobName) throws IOException {
                maybeIOExceptionOrBlock(blobName);
                if (blobName.startsWith(BlobStoreRepository.SNAPSHOT_PREFIX) && path().equals(basePath()) == false) {
                    if (blockOnWriteShardLevelMeta) {
                        blockExecutionAndMaybeWait(blobName);
                    } else if (blockAndFailOnWriteShardLevelMeta) {
                        blockExecutionAndFail(blobName);
                    }
                }
            }

            @Override
            public void writeBlobAtomic(
                OperationPurpose purpose,
                String blobName,
                InputStream inputStream,
                long blobSize,
                boolean failIfAlreadyExists
            ) throws IOException {
                final Random random = beforeAtomicWrite(blobName);
                if ((delegate() instanceof FsBlobContainer) && (random.nextBoolean())) {
                    // Simulate a failure between the write and move operation in FsBlobContainer
                    final String tempBlobName = FsBlobContainer.tempBlobName(blobName);
                    super.writeBlob(purpose, tempBlobName, inputStream, blobSize, failIfAlreadyExists);
                    maybeIOExceptionOrBlock(blobName);
                    final FsBlobContainer fsBlobContainer = (FsBlobContainer) delegate();
                    fsBlobContainer.moveBlobAtomic(purpose, tempBlobName, blobName, failIfAlreadyExists);
                } else {
                    // Atomic write since it is potentially supported
                    // by the delegating blob container
                    maybeIOExceptionOrBlock(blobName);
                    super.writeBlobAtomic(purpose, blobName, inputStream, blobSize, failIfAlreadyExists);
                }
            }

            @Override
            public void writeBlobAtomic(
                final OperationPurpose purpose,
                final String blobName,
                final BytesReference bytes,
                final boolean failIfAlreadyExists
            ) throws IOException {
                writeBlobAtomic(purpose, blobName, bytes.streamInput(), bytes.length(), failIfAlreadyExists);
            }

            private Random beforeAtomicWrite(String blobName) throws IOException {
                final Random random = RandomizedContext.current().getRandom();
                if (failOnIndexLatest && BlobStoreRepository.INDEX_LATEST_BLOB.equals(blobName)) {
                    throw new IOException("Random IOException");
                }
                if (shouldFail(blobName, randomControlIOExceptionRate)) {
                    logger.info("throwing random IOException for atomic write of file [{}] at path [{}]", blobName, path());
                    throw new IOException("Random IOException");
                }
                if (blobName.startsWith(BlobStoreRepository.INDEX_FILE_PREFIX)) {
                    if (blockAndFailOnWriteIndexFile) {
                        blockExecutionAndFail(blobName);
                    } else if (blockOnWriteIndexFile) {
                        blockExecutionAndMaybeWait(blobName);
                    }
                }
                return random;
            }
        }
    }
}
