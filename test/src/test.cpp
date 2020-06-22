#include <gtest/gtest.h>
#include <arailib.hpp>
#include <dbscan.hpp>

using namespace arailib;
using namespace dbscan;

TEST(dbscan, small) {
    string data_path = "/tmp/tmp.VzjXG3U2nU/test/src/data1.csv";
    double eps = 1.5;
    int minpts = 2;

    auto dbscan = DBSCAN(eps, minpts);
    dbscan.fit(data_path);

    ASSERT_EQ(dbscan.cluster.size(), 2);
    ASSERT_EQ(dbscan.database[0].cluster_id, 0);
    ASSERT_EQ(dbscan.database[1].cluster_id, 1);
    ASSERT_EQ(dbscan.database[10].cluster_id, -1);
}

TEST(dbscan, moon) {
    string data_path = "/home/arai/workspace/dataset/cluster/moon.csv";
    string save_path = "/home/arai/workspace/result/clustering/dbscan/moon.csv";

    double eps = 0.2;
    int minpts = 5;

    auto dbscan = DBSCAN(eps, minpts);
    dbscan.fit(data_path);

    ASSERT_EQ(dbscan.cluster.size(), 2);
    ASSERT_EQ(dbscan.cluster[0].size(), 100);
    ASSERT_EQ(dbscan.cluster[1].size(), 99);

    dbscan.save(save_path);
}

TEST(dbscan, small_with_graph) {
    string base_dir = "/tmp/tmp.VzjXG3U2nU/test/src/";
    string data_path = base_dir + "data1.csv";
    string graph_path = base_dir + "graph1.csv";

    const auto dataset = load_data(data_path, -1);

    double eps = 2.5;
    int minpts = 3;

    GraphIndex graph;
    graph.load(data_path, graph_path);

    vector<vector<int>> eps_neighbors_list;
    for (const auto node : graph.nodes) {
        const auto eps_neighbor = graph.self_range_search(node.data.id, eps);
        eps_neighbors_list.emplace_back(eps_neighbor);
    }

    ASSERT_EQ(eps_neighbors_list[0].size(), 4);

    auto dbscan = DBSCAN(eps, minpts);
    dbscan.fit(dataset, eps_neighbors_list);

    ASSERT_EQ(dbscan.cluster.size(), 2);
    ASSERT_EQ(dbscan.database[0].cluster_id, 0);
    ASSERT_EQ(dbscan.database[1].cluster_id, 1);
    ASSERT_EQ(dbscan.database[10].cluster_id, -1);
}