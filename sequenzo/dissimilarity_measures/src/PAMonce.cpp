#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#define WEIGHTED_CLUST_TOL -1e-10

// 加入了判断平台的宏
// MSVC 下通过 basetsd.h 引入了 SSIZE_T 并定义了 ssize_t
// 非 Windows（Linux/macOS）下继续用 <unistd.h>（它原生就有 ssize_t）
// 如果是 Windows (_MSC_VER)，如果没有定义过 ssize_t，那就定义它；
// 否则（Linux/macOS），照常 #include <unistd.h>
#if defined(_MSC_VER)
    #include <basetsd.h>
    #ifndef _SSIZE_T_DEFINED
        typedef SSIZE_T ssize_t;
        #define _SSIZE_T_DEFINED
    #endif
#else
    #include <unistd.h>
#endif

namespace py = pybind11;

class PAMonce {
public:
    PAMonce(int nelement, py::array_t<double> diss, py::array_t<int> centroids, int npass, py::array_t<double> weights){
        py::print("[>] PAMonce starts ... ");
        std::cout << std::flush;

        try {
            this->nelement = nelement;
            this->diss = diss;
            this->centroids = centroids;
            this->npass = npass;
            this->weights = weights;

            clusterid = py::array_t<int>(nelement);
            tclusterid.resize(nelement, -1);

            maxdist = find_max_value(diss);
            dysma.resize(nelement, maxdist);
            dysmb.resize(nelement, maxdist);

            fvect.resize(nelement, 0);
            nclusters = centroids.size();
        } catch (const std::exception& e){
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    double find_max_value(py::array_t<double> diss) {
        // 获取数组维度
        py::buffer_info buf_info = diss.request();
        double *ptr = static_cast<double *>(buf_info.ptr);

        ssize_t rows = buf_info.shape[0];
        ssize_t cols = buf_info.shape[1];

        double max_val = -std::numeric_limits<double>::infinity(); // 初始化为负无穷

        for (ssize_t i = 0; i < rows; i++) {
            for (ssize_t j = 0; j < cols; j++) {
                double val = ptr[i * cols + j];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }

        return max_val;
    }

    py::array_t<int> runclusterloop(){
        auto ptr_diss = diss.unchecked<2>();
        auto ptr_weights = weights.unchecked<1>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        auto ptr_clusterid = clusterid.mutable_unchecked<1>();

        for(int i=0; i<nelement; i++){
            ptr_clusterid(i) = -1;
        }

        double dzsky = 1;
        int hbest = -1, nbest = -1;
        double total = -1;

        do {
            for(int i=0; i<nelement; i++){
                for(int k=0; k<nclusters; k++){
                    int i_cluster = ptr_centroids(k);

                    if(dysma[i] > ptr_diss(i, i_cluster)){
                        dysmb[i] = dysma[i];
                        dysma[i] = ptr_diss(i, i_cluster);
                        tclusterid[i] = k;
                    }else if(dysmb[i] > ptr_diss(i, i_cluster)){
                        dysmb[i] = ptr_diss(i, i_cluster);
                    }
                }
            }

            if(total < 0){
                total = 0;
                for(int i=0; i<nelement; i++){
                    total += ptr_weights(i) * dysma[i];
                }
            }

            dzsky = 1;

            for(int k=0; k<nclusters; k++){
                int i = ptr_centroids(k);
                double removeCost = 0;

                for(int j=0; j<nelement; j++){
                    if(tclusterid[j] == k){
                        removeCost += ptr_weights(j) * (dysmb[j] - dysma[j]);
                        fvect[j] = dysmb[j];
                    }else{
                        fvect[j] = dysma[j];
                    }
                }

                //Now check possible new medoids h
                for(int h=0; h<nelement; h++){
                    if(ptr_diss(h, i) > 0){
                        double addGain = removeCost;

                        //Compute gain of adding h as a medoid
                        for(int j=0; j<nelement; j++){
                            if(ptr_diss(h, j) < fvect[j]){
                                addGain += ptr_weights(j) * (ptr_diss(h, j) - fvect[j]);
                            }
                        }

                        if(dzsky > addGain){
                            dzsky = addGain;
                            hbest = h;
                            nbest = i;
                        }
                    }
                }
            }

            if(dzsky < WEIGHTED_CLUST_TOL){
                for(int k=0; k<nclusters; k++){
                    if(ptr_centroids(k) == nbest){
                        ptr_centroids(k) = hbest;
                    }
                }

                total += dzsky;
            }
        } while (dzsky < WEIGHTED_CLUST_TOL);

        for(int j=0; j<nelement; j++){
            ptr_clusterid(j) = ptr_centroids[tclusterid[j]];
        }

        return clusterid;
    }

private:
    int nelement;
    py::array_t<double> diss;
    py::array_t<int> centroids;
    int npass;
    py::array_t<double> weights;

    py::array_t<int> clusterid;
    std::vector<int> tclusterid;

    double maxdist;
    std::vector<double> dysma;
    std::vector<double> dysmb;

    std::vector<double> fvect;
    int nclusters;
};