import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_batsmen_performance(batsmen_performance_df: pd.DataFrame) -> pd.DataFrame:

    """Preprocesses the data for batsmen.

    Args:
        batsmen_performance_df: Processed data of batsman as dataframe.
    Returns:
        Preprocessed data, with class assigned to each batsman
        returns a dataframe
    """
    scaler = StandardScaler()

    batsmen_performance_df['batting_average'] = scaler.fit_transform(batsmen_performance_df[['batting_average']])
    batsmen_performance_df['strike_rate'] = scaler.fit_transform(batsmen_performance_df[['strike_rate']])
    batsmen_performance_df['half_centuries'] = scaler.fit_transform(batsmen_performance_df[['half_centuries']])
    batsmen_performance_df['centuries'] = scaler.fit_transform(batsmen_performance_df[['centuries']])
    batsmen_performance_df['boundary_score_percentage'] = scaler.fit_transform(batsmen_performance_df[['boundary_score_percentage']])
    batsmen_performance_df['consistance'] = scaler.fit_transform(batsmen_performance_df[['consistance']])
    batsmen_performance_df['score_25_percentile'] = scaler.fit_transform(batsmen_performance_df[['score_25_percentile']])
    batsmen_performance_df['score_50_percentile'] = scaler.fit_transform(batsmen_performance_df[['score_50_percentile']])
    batsmen_performance_df['score_75_percentile'] = scaler.fit_transform(batsmen_performance_df[['score_75_percentile']])


    batsmen_batting_average_percentiles = np.percentile(batsmen_performance_df['batting_average'], [25, 50, 75])
    batsmen_strike_rate_percentiles = np.nanpercentile(batsmen_performance_df['strike_rate'], [25, 50, 75])
    batsmen_consistance_percentiles = np.nanpercentile(batsmen_performance_df['consistance'], [25, 50, 75])

    batsmen_performance_df.loc[(batsmen_performance_df['batting_average'] <= batsmen_batting_average_percentiles[0]) &
                           (batsmen_performance_df['strike_rate'] <= batsmen_strike_rate_percentiles[0]) &
                           (batsmen_performance_df['consistance'] <= batsmen_consistance_percentiles[0]), 'class'] = 0

    batsmen_performance_df.loc[(batsmen_performance_df['batting_average'] > batsmen_batting_average_percentiles[0]) & 
                            (batsmen_performance_df['batting_average'] <= batsmen_batting_average_percentiles[1]) &
                            (batsmen_performance_df['strike_rate'] > batsmen_strike_rate_percentiles[0]) & 
                            (batsmen_performance_df['strike_rate'] <= batsmen_strike_rate_percentiles[1]) &
                            (batsmen_performance_df['consistance'] > batsmen_consistance_percentiles[0]) & 
                            (batsmen_performance_df['consistance'] <= batsmen_consistance_percentiles[1]), 'class'] = 1

    batsmen_performance_df.loc[(batsmen_performance_df['batting_average'] >= batsmen_batting_average_percentiles[2]) &
                            (batsmen_performance_df['strike_rate'] >= batsmen_strike_rate_percentiles[2]) &
                            (batsmen_performance_df['consistance'] >= batsmen_consistance_percentiles[2]), 'class'] = 2

    batsmen_performance_df = batsmen_performance_df.drop(columns= ['batsman_striker_name','batter_id'])

    return batsmen_performance_df

def preprocess_bowler_performance(bowler_performance_df: pd.DataFrame) -> pd.DataFrame:

    """Preprocesses the data for bowler.

    Args:
        bowler_performance_df: Processed data of bowler as dataframe.
    Returns:
        Preprocessed data, with class assigned to each bowler
        returns a dataframe
    """
    scaler = StandardScaler()

    bowler_performance_df['bowling_average'] = scaler.fit_transform(bowler_performance_df[['bowling_average']])
    bowler_performance_df['bowler_economy_rate'] = scaler.fit_transform(bowler_performance_df[['bowler_economy_rate']])
    bowler_performance_df['bowler_strike_rate'] = scaler.fit_transform(bowler_performance_df[['bowler_strike_rate']])
    bowler_performance_df['dot_ball_percentage'] = scaler.fit_transform(bowler_performance_df[['dot_ball_percentage']])
    bowler_performance_df['wickets_caught'] = scaler.fit_transform(bowler_performance_df[['wickets_caught']])
    bowler_performance_df['wickets_bowled'] = scaler.fit_transform(bowler_performance_df[['wickets_bowled']])
    bowler_performance_df['wickets_lbw'] = scaler.fit_transform(bowler_performance_df[['wickets_lbw']])

    bowling_average_percentiles = np.percentile(bowler_performance_df['bowling_average'], [25, 50, 75])
    bowling_strike_rate_percentiles = np.nanpercentile(bowler_performance_df['bowler_strike_rate'], [25, 50, 75])
    bowling_economy_rate_percentiles = np.nanpercentile(bowler_performance_df['bowler_economy_rate'], [25, 50, 75])

    bowler_performance_df.loc[(bowler_performance_df['bowling_average'] <= bowling_average_percentiles[0]) &
                            (bowler_performance_df['bowler_strike_rate'] <= bowling_strike_rate_percentiles[0]) &
                            (bowler_performance_df['bowler_economy_rate'] <= bowling_economy_rate_percentiles[0]), 'class'] = 0

    bowler_performance_df.loc[(bowler_performance_df['bowling_average'] > bowling_average_percentiles[0]) & 
                            (bowler_performance_df['bowling_average'] <= bowling_average_percentiles[1]) &
                            (bowler_performance_df['bowler_strike_rate'] > bowling_strike_rate_percentiles[0]) & 
                            (bowler_performance_df['bowler_strike_rate'] <= bowling_strike_rate_percentiles[1]) &
                            (bowler_performance_df['bowler_economy_rate'] > bowling_economy_rate_percentiles[0]) & 
                            (bowler_performance_df['bowler_economy_rate'] <= bowling_economy_rate_percentiles[1]), 'class'] = 1

    bowler_performance_df.loc[(bowler_performance_df['bowling_average'] >= bowling_average_percentiles[2]) &
                            (bowler_performance_df['bowler_strike_rate'] >= bowling_strike_rate_percentiles[2]) &
                            (bowler_performance_df['bowler_economy_rate'] >= bowling_economy_rate_percentiles[2]), 'class'] = 2


    return bowler_performance_df

def create_batsmen_model_input_table(
    batsmen_performance: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        batsmen performance data: Preprocessed data of batsmen.

    Returns:
        batsmen model input table.

    """

    batsmen_model_input_table = batsmen_performance.dropna()
    return batsmen_model_input_table

def create_bowler_model_input_table(
    bowler_performance: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        bowler performance data: Preprocessed data of batsmen.

    Returns:
        bowler model input table.

    """

    bowler_model_input_table = bowler_performance.dropna()
    return bowler_model_input_table
