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