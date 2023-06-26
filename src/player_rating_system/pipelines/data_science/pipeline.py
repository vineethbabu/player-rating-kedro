from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["batsmen_model_input_table", "params:batsmen_model_options"],
                outputs=["X_train_batsmen", "X_test_batsmen", "y_train_batsmen", "y_test_batsmen"],
                name="batsmen_split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_batsmen", "y_train_batsmen"],
                outputs="classifier_batsmen",
                name="batsmen_train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier_batsmen", "X_test_batsmen", "y_test_batsmen"],
                outputs=None,
                name="batsmen_evaluate_model_node",
            ),
            node(
                func=split_data,
                inputs=["bowler_model_input_table", "params:bowler_model_options"],
                outputs=["X_train_bowler", "X_test_bowler", "y_train_bowler", "y_test_bowler"],
                name="bowler_split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_bowler", "y_train_bowler"],
                outputs="classifier_bowler",
                name="bowler_train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier_bowler", "X_test_bowler", "y_test_bowler"],
                outputs=None,
                name="bowler_evaluate_model_node",
            ),
        ]
    )