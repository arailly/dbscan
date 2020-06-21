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
        Database result;
        for (const auto& data : dataset) {
            result.emplace_back(Point(data));
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
                if (point_id == data.id) continue;

                const auto dist = euclidean_distance(query, data);
                if (dist < eps)
                    result.emplace_back(data.id);
            }
            return result;
        }

        void assign(Point& point, int cluster_id) {
            point.cluster_id = cluster_id;
            cluster[cluster_id].emplace_back(point.id);
        }

        void expand_cluster(Point& point, vector<int> neighbors, int cluster_id,
                            const vector<vector<int>>& eps_neighbors_list,
                            unordered_map<int, bool>& visited) {
            // assign cluster
            assign(point, cluster_id);

            for (int i = 0; i < neighbors.size(); ++i) {
                const auto neighbor_id = neighbors[i];
                auto& neighbor = database[neighbor_id];
                if (!visited[neighbor_id]) {
                    visited[neighbor_id] = true;

                    const auto& neighbor_neighbors = eps_neighbors_list[neighbor_id];
                    if (neighbor_neighbors.size() >= minpts) {
                        for (const auto neighbor_neighbor_id : neighbor_neighbors)
                            neighbors.emplace_back(neighbor_neighbor_id);
                    }
                }
                if (neighbor.cluster_id < 0) {
                    assign(neighbor, cluster_id);
                }
            }
        }

        // cite from https://ja.wikipedia.org/wiki/DBSCAN
        void fit(string data_path, int n = -1) {
            database = convert(load_data(data_path, n));

            // calculate eps neighbors
            vector<vector<int>> eps_neighbors_list(database.size());
#pragma omp parallel for simd
            for (int id = 0; id < database.size(); ++id)
                eps_neighbors_list[id] = scan_eps_neighbors(id);

            // clustering
            int cluster_id = -1;
            unordered_map<int, bool> visited;

            for (auto& point : database) {
                if (visited[point.id]) continue;
                visited[point.id] = true;

                const auto& eps_neighbors = eps_neighbors_list[point.id];
                if (eps_neighbors.size() < minpts)
                    point.cluster_id = -1; // noise
                else {
                    ++cluster_id;
                    cluster.resize(cluster_id + 1);
                    expand_cluster(point, eps_neighbors, cluster_id,
                                   eps_neighbors_list, visited);
                }
            }
        }

        void save(string save_path) {
            ofstream ofs(save_path);
            ofs << "cluster_id" << endl;
            for (const auto& point : database) {
                ofs << point.cluster_id << endl;
            }
        }
    };
}

#endif //DBSCAN_DBSCAN_HPP
