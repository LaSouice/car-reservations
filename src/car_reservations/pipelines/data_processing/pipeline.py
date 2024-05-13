from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_reservations, preprocess_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
           node(
                func=preprocess_reservations,
                inputs="reservations",
                outputs="preprocessed_reservations",
                name="preprocessed_reservations_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_reservations", "vehicles"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
            node(
                func=preprocess_model_input_table,
                inputs="model_input_table",
                outputs="preprocessed_model_input_table",
                name="preprocess_model_input_table_node",
            ),
        ]
    )
