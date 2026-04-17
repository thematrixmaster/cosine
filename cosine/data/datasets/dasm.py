from netam.common import parallel_df_apply
from netam.sequences import assert_pcp_valid
from tqdm import tqdm

# Provides pd.DataFrame.progress_apply:
tqdm.pandas()


class _ValidPCPFilterFunc:

    def __init__(self, parent_func, child_func):
        self.parent_func = parent_func
        self.child_func = child_func

    def __call__(self, row):
        try:
            # Although this is not exactly how heavy and light sequences are combined
            # for model application, it should work for checking which ones are valid.
            assert_pcp_valid(
                self.parent_func(row),
                self.child_func(row),
            )
            return True
        except ValueError:
            return False


def _default_get_row_parent(row):
    return row["parent_heavy"][: (len(row["parent_heavy"]) // 3) * 3] + row["parent_light"]


def _default_get_row_child(row):
    return row["child_heavy"][: (len(row["parent_heavy"]) // 3) * 3] + row["child_light"]


def filter_valid_pcps(
    pcp_df,
    get_row_parent=_default_get_row_parent,
    get_row_child=_default_get_row_child,
    parallelize=True,
    force_parallel=None,
):
    """Filter out PCPs whose ambiguities make them unusable for inference.

    Args:
        pcp_df: A DataFrame containing the pcp data.
        get_row_parent: A function that takes a row and returns the parent sequence.
        get_row_child: A function that takes a row and returns the child sequence.
        parallelize: If True, use parallel processing to filter the DataFrame. If there are too
            few rows or too few processors, the call will not be parallelized regardless of this value.
        force_parallel: If an integer is provided, it will force the use of that many parallel processes, regardless of the value of parallelize.


    In order to run this function in parallel, the functions passed to
    get_row_parent and get_row_child must be pickleable.

    Modifies the passed pcp_df in-place and returns it.
    """
    print("Filtering PCPs without mutations, or whose ambiguities make them unusable for inference")

    filter_func = _ValidPCPFilterFunc(get_row_parent, get_row_child)

    if parallelize or force_parallel is not None:
        pcp_df = pcp_df[
            parallel_df_apply(
                pcp_df,
                filter_func,
                force_parallel=force_parallel,
                use_progress_apply=True,
            )
        ]
    else:
        pcp_df = pcp_df[pcp_df.progress_apply(filter_func, axis=1)]
    return pcp_df
