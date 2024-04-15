#include "rcpp_asap.hh"

#include "rcpp_asap_mtx_data.hh"

#ifndef RCPP_ASAP_MTX_MATCHING_DATA_HH_
#define RCPP_ASAP_MTX_MATCHING_DATA_HH_

struct mtx_matching_data_t {

    struct MTX {
        MTX(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    struct ROW {
        ROW(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    struct COL {
        COL(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    struct IDX {
        IDX(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    // explicit mtx_data_t(const MTX &mtx1,
    //                     const ROW &row1,
    //                     const ROW &col1,
    //                     const IDX &idx1,
    //                     const std::size_t MAX_ROW_WORD = 2,
    //                     const char ROW_WORD_SEP = '_')
    //     : mtx_file(mtx.val)
    //     , row_file(row.val)
    //     , idx_file(idx.val)
    // {
    // }
};

#endif // end of mtx MATCHING data
