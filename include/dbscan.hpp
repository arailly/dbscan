//
// Created by Yusuke Arai on 2020/06/21.
//

#ifndef DBSCAN_DBSCAN_HPP
#define DBSCAN_DBSCAN_HPP

#include <arailib.hpp>

using namespace arailib;

namespace dbscan {
    struct Point : public Data<> {
        int cluster_id;
        Point(Data<> data) : Data<>(data.id, data.x), cluster_id(-1) {}
    };

    using Database = vector<Point>;

    // arailib::Dataset -> dbscan::Database
    auto convert(Dataset<> dataset) {
        Database result(dataset.size());
#pragma omp parallel for
        for (int id = 0; id < dataset.size(); ++id) {
            const auto& data = dataset[id];
            result[id] = Point(data);
        }
        return result;
    }

    using Cluster = vector<vector<int>>;

    struct DBSCAN {
        double eps;
        int minpts;
        Database database;
        Cluster cluster;

        DBSCAN(double eps, int minpts) : eps(eps), minpts(minpts) {}

        auto scan_eps_neighbors(int point_id) {
            const auto& query = database[point_id];
            vector<int> result;
            for (const auto data : database) {
                const auto dist = euclidean_distance(query, data);
                if (dist < eps)
                    result.emplace_back(data.id);
            }
            return result;
        }

        void fit(string data_path, int n = -1) {
            database = convert(load_data(data_path, n));

            // calculate eps neighbors
            vector<vector<int>> eps_neighbors_list(database.size());
#pragma omp parallel for simd
            for (int id = 0; id < database.size(); ++id)
                eps_neighbors_list[id] = scan_eps_neighbors(id);

            // assign cluster
            int cluster_id = 0;
            unordered_map<int, bool> visited;

            for (auto& point : database) {
                if (visited[point.id]) continue;
                visited[point.id] = true;

                const auto& eps_neighbors = eps_neighbors_list[point.id];
                if (eps_neighbors.size() < minpts)
                    point.cluster_id = -1; // noise
                else {
                    ++cluster_id;
                }
            }
        }
    };
}

#endif //DBSCAN_DBSCAN_HPP
