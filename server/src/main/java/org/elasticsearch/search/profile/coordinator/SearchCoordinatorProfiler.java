/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile.coordinator;

import org.elasticsearch.search.profile.AbstractProfileBreakdown;
import org.elasticsearch.search.profile.SearchProfileCoordinatorResults;

public class SearchCoordinatorProfiler extends AbstractProfileBreakdown<SearchCoordinatorTimingType> {

    private final String nodeId;

    public SearchCoordinatorProfiler(final String nodeId) {
        super(SearchCoordinatorTimingType.class);
        this.nodeId = nodeId;
    }

    public String getNodeId() {
        return nodeId;
    }

    public SearchProfileCoordinatorResults build() {
        return new SearchProfileCoordinatorResults(nodeId, toBreakdownMap());
    }
}
