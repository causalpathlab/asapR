#include "rcpp_mtx_data.hh"

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

    explicit mtx_matching_data_t(const mtx_tuple_t &lhs,
                                 const mtx_tuple_t &rhs,
                                 const std::size_t MAX_ROW_WORD = 2,
                                 const char ROW_WORD_SEP = '_')
        : lhs_mtx_file(lhs.mtx.val)
        , lhs_row_file(lhs.row.val)
        , lhs_idx_file(lhs.idx.val)
        , rhs_mtx_file(rhs.mtx.val)
        , rhs_row_file(rhs.row.val)
        , rhs_idx_file(rhs.idx.val)
    {
    }

    Index lhs_max_row() const;
    Index lhs_max_col() const;
    Index lhs_max_elem() const;

    Index rhs_max_row() const;
    Index rhs_max_col() const;
    Index rhs_max_elem() const;

public:
    const std::string lhs_mtx_file;
    const std::string lhs_row_file;
    const std::string lhs_idx_file;

    const std::string rhs_mtx_file;
    const std::string rhs_row_file;
    const std::string rhs_idx_file;

private:
    mm_info_reader_t lhs_info;
    mm_info_reader_t rhs_info;

    std::vector<Index> lhs_mtx_idx;
    std::vector<Index> rhs_mtx_idx;
    std::vector<std::string> sub_rows;
};

#endif // end of mtx MATCHING data
