/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.retriever;

import org.elasticsearch.index.query.TermQueryBuilder;
import org.elasticsearch.search.rank.TestRankDoc;
import org.elasticsearch.test.ESTestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.elasticsearch.search.profile.query.RandomQueryGenerator.randomQueryBuilder;

public class RankDocsRetrieverBuilderTests extends ESTestCase {

    private Supplier<TestRankDoc[]> rankDocsSupplier() {
        final int rankDocsCount = randomIntBetween(0, 10);
        final int shardIndex = 0;
        TestRankDoc[] rankDocs = new TestRankDoc[rankDocsCount];
        int docId = 0;
        for (int i = 0; i < rankDocsCount; i++) {
            TestRankDoc testRankDoc = new TestRankDoc(docId, randomFloat(), shardIndex);
            docId += randomInt(100);
        }
        return () -> rankDocs;
    }

    public void testWithStandardRetriever() {
        final List<String> textFields = randomList(2, () -> randomAlphaOfLengthBetween(1, 10));
        final List<String> numericFields = randomList(2, () -> randomAlphaOfLengthBetween(1, 10));

        StandardRetrieverBuilder standardRetriever = new StandardRetrieverBuilder();
        standardRetriever.queryBuilder = randomQueryBuilder(textFields, numericFields, 100, 5);
        standardRetriever.retrieverName = "standardRetriever";
        standardRetriever.preFilterQueryBuilders = new ArrayList<>();
        if (randomBoolean()) {
            for (int i = 0; i < randomInt(2); i++) {
                standardRetriever.preFilterQueryBuilders.add(new TermQueryBuilder("field", "value" + i));
            }
        }
    }
}
