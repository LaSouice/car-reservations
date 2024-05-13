from typing import Dict, Tuple
import logging
import pandas as pd


def preprocess_reservations(reservations: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for reservations.

    Args:
        reservations: Raw data.
    Returns:
        Preprocessed data, group by vehicles_id and count the number of reservations for each vehicle.
    """
    
    return reservations.groupby('vehicle_id').size().reset_index(name='reservations')


def create_model_input_table(
    vehicles: pd.DataFrame, preprocessed_reservations: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        vehicles: Table of all vehicles and their attributes
        reservations: Table of all completed reservations
    Returns:
        Model input table.

    """
    model_input_table = pd.merge(vehicles, preprocessed_reservations, on='vehicle_id', how='left')
    model_input_table['reservations'] = model_input_table['reservations'].astype(int)
    return model_input_table


def preprocess_model_input_table(model_input_table: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for model_input_table.

    Args:
        model_input_table: Model input table.
    Returns:
        Preprocessed Model input table. replace 'reservations' null values with 0, 
        Drop rows with # of reservations < 7,
        Drop column 'vehicle_id'.
    """
    model_input_table = model_input_table.fillna(0)

    reservations_count = model_input_table['reservations'].value_counts().sort_values()
    model_input_table = model_input_table[model_input_table['reservations'].map(reservations_count) > 7]
    model_input_table = model_input_table.drop('vehicle_id', axis=1)
    return model_input_table