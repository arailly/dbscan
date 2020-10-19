# DBSCAN
## Overview
It is implementation of clustering algorithm, DBSCAN: Density-Based Spatial Clustering of Applications with Noise.

The main algorithm is written `include/dbscan.hpp` (header only), so you can use it easily.

Reference: A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (M. Ester et al., KDD 1996)

## Example
```
#include <arailib.hpp>
#include <dbscan.hpp>

using namespace std;
using namespace arailib;
using namespace dbscan;

int main() {
    const string data_path = "path/to/data.csv";
    const unsigned n = 1000; // data size
    const float eps = 1.0; // epsilon
    const int minpts = 5; // minpts

    auto dbscan = DBSCAN(eps, minpts);
    dbscan.fit(data_path, n);
}
```

## Input File Format
If you want to try clustering with this three vectors, `(0, 1), (2, 4), (3, 3)`, you must describe data.csv like following format:
```
0,1
2,4
3,3
```
