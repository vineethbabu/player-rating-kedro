from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_batsmen_performance, preprocess_bowler_performance, create_batsmen_model_input_table, create_bowler_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_batsmen_performance,
                inputs="batsmandata",
                outputs="preprocessed_batsmen_performance",
                name="preprocess_batsmen_node",
            ),
            node(
                func=preprocess_bowler_performance,
                inputs="bowlerdata",
                outputs="preprocessed_bowler_performance",
                name="preprocess_bowler_node",
            ),
            node(
                func=create_batsmen_model_input_table,
                inputs=["preprocessed_batsmen_performance"],
                outputs="batsmen_model_input_table",
                name="create_batsmen_model_input_table_node",
            ),
            node(
                func=create_bowler_model_input_table,
                inputs=["preprocessed_bowler_performance"],
                outputs="bowler_model_input_table",
                name="create_bowler_model_input_table_node",
            ),
        ]
    )