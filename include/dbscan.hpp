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

    using Clusters = vector<vector<int>>;

    struct Node {
        const Data<> data;
        vector<int> neighbors;
        unordered_map<size_t, bool> added;

        void init() { added[data.id] = true; }
        Node() : data(Data<>(0, {0})) { init(); }
        Node(Data<>& p) : data(move(p)) { init(); }

        void add_neighbor(int node_id) {
            if (added.find(node_id) != added.end()) return;
            added[node_id] = true;
            neighbors.emplace_back(node_id);
        }

        void clear_neighbor() {
            neighbors.clear();
            added.clear();
            added[data.id] = true;
        }

        auto get_n_neighbors() const { return neighbors.size(); }
    };

    struct GraphIndex {
        vector<Node> nodes;

        auto size() const { return nodes.size(); }

        auto empty() const { return nodes.empty(); }

        auto begin() const { return nodes.begin(); }

        auto end() const { return nodes.end(); }

        auto& operator[](size_t i) { return nodes[i]; }

        auto& operator[](const Node &n) { return nodes[n.data.id]; }

        const auto& operator[](size_t i) const { return nodes[i]; }

        const auto& operator[](const Node &n) const { return nodes[n.data.id]; }

        void init_data(Series<>& series) {
            for (auto &point : series) {
                nodes.emplace_back(point);
            }
        }

        void make_bidirectional() {
            for (auto &node : nodes) {
                for (const auto neighbor_id : node.neighbors) {
                    nodes[neighbor_id].add_neighbor(node.data.id);
                }
            }
        }

        void load(Series<>& series, const string& graph_path, int n, int degree = -1) {
            init_data(series);

            // csv file
            if (is_csv(graph_path)) {
                ifstream ifs(graph_path);
                if (!ifs) {
                    const string message = "Can't open file!: " + graph_path;
                    throw runtime_error(message);
                }

                string line;
                while (getline(ifs, line)) {
                    const auto ids = split<size_t>(line);
                    nodes[ids[0]].add_neighbor(ids[1]);
                    if (degree != -1 && nodes[ids[0]].get_n_neighbors() >= degree) break;
                }
                return;
            }

            // dir
#pragma omp parallel
            {
#pragma omp for schedule(dynamic, 10) nowait
                for (int i = 0; i < n; i++) {
                    const string path = graph_path + "/" + to_string(i) + ".csv";
                    ifstream ifs(path);

                    if (!ifs) {
                        const string message = "Can't open file!: " + path;
                        throw runtime_error(message);
                    }

                    string line;
                    while (getline(ifs, line)) {
                        const auto ids = split<size_t>(line);
                        if (degree == -1 || nodes[ids[0]].get_n_neighbors() < degree) {
                            nodes[ids[0]].add_neighbor(ids[1]);
                        }
                    }
                }
            };
        }

        void load(const string& data_path, const string& graph_path,
                  int n = -1, int degree = -1) {
            auto series = load_data(data_path, n);
            load(series, graph_path, n, degree);
        }

        virtual void save(const string &save_path) {
            // csv
            if (is_csv(save_path)) {
                ofstream ofs(save_path);
                string line;
                for (const auto &node : nodes) {
                    line = to_string(node.data.id);
                    for (const auto &neighbor_id : node.neighbors) {
                        line += ',' + to_string(neighbor_id);
                    }
                    line += '\n';
                    ofs << line;
                }
                return;
            }

            // dir
            vector<string> lines(static_cast<unsigned long>(ceil(nodes.size() / 1000.0)));
            for (const auto& node : nodes) {
                const size_t line_i = node.data.id / 1000;
                for (const auto& neighbor_id : node.neighbors) {
                    lines[line_i] += to_string(node.data.id) + "," +
                                     to_string(neighbor_id) + "\n";
                }
            }

            for (int i = 0; i < lines.size(); i++) {
                const string path = save_path + "/" + to_string(i) + ".csv";
                ofstream ofs(path);
                ofs << lines[i];
            }
        }

        auto self_range_search(int query_id, float range) const {
            const auto& query_node = nodes[query_id];

            unordered_map<int, bool> added;
            added[query_id] = true;

            unordered_map<int, bool> checked;

            vector<int> result;
            result.emplace_back(query_id);

            while (true) {
                int first_unchecked_id = 0;
                bool is_over = true;
                for (const auto& e : result) {
                    if (checked[e]) continue;
                    checked[e] = true;
                    first_unchecked_id = e;
                    is_over = false;
                    break;
                }

                if (is_over) {
                    result.erase(result.begin());
                    return result;
                }

                const auto& first_unchecked_node = nodes[first_unchecked_id];
                for (const auto& neighbor_id : first_unchecked_node.neighbors) {
                    if (added[neighbor_id]) continue;
                    added[neighbor_id] = true;

                    const auto& neighbor = nodes[neighbor_id];
                    const auto dist = euclidean_distance(query_node.data, neighbor.data);
                    if (dist < range) result.emplace_back(neighbor_id);
                }
            }
        }
    };

    struct DBSCAN {
        double eps;
        int minpts;
        Database database;
        Clusters clusters;

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
            clusters[cluster_id].emplace_back(point.id);
        }

        void expand_cluster(Point& point, vector<int> neighbors, int cluster_id,
                            const vector<vector<int>>& eps_neighbors_list,
                            unordered_map<int, bool>& visited) {
            // assign clusters
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
        void fit(const Dataset<>& dataset,
                 vector<vector<int>> eps_neighbors_list = vector<vector<int>>()) {
            database = convert(dataset);

            // calculate eps neighbors if eps_neighbors_list is empty
            if (eps_neighbors_list.empty()) {
                eps_neighbors_list.resize(database.size());
#pragma omp parallel for simd
                for (int id = 0; id < database.size(); ++id)
                    eps_neighbors_list[id] = scan_eps_neighbors(id);
            }

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
                    clusters.resize(cluster_id + 1);
                    expand_cluster(point, eps_neighbors, cluster_id,
                                   eps_neighbors_list, visited);
                }
            }
        }

        void fit(string data_path, int n = -1) {
            const auto dataset = load_data(data_path, n);
            fit(dataset);
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
