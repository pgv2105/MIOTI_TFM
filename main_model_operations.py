import pandas as pd
import pickle

from models_operations.models import evaluate_with_LSTM
from services_preprocessing.preprocess_LSTM_data import preprocess_LSTM_data

if __name__ == '__main__':

    X_LSTM = pd.read_pickle('extraction/INPUT_2022_2025_EXTENDED.pkl')
    Y_LSTM = pd.read_pickle('extraction/OUTPUT_2022_2025_EXTENDED.pkl')

    X, Y = preprocess_LSTM_data(X_LSTM, Y_LSTM)

    # Load previous preprocessed data
    with open("training_data_model_PLUS_MULTIVARIATE.pkl", "rb") as f:
        data = pickle.load(f)

    X_model = data["dataset_X"]
    Y_model = data["dataset_Y"]

    # TODO: Change later with statistical report (VIEWED IN CLASS 1 STATISTICS)
    # evaluate_dataset(X_LSTM,Y_LSTM)

    evaluate_with_LSTM(X_model,Y_model)
























    ###########################   CRYPTO   #############################
    #crypto = 'ETH'
    # Define the contract with primary exchange
    #contract = create_contract_crypto(crypto)
    #print(f"STOCK = {crypto} ")
    # Request historical data
    #client.reqHistoricalData(
    #    1, contract, '', '3600 S', '30 secs', 'AGGTRADES', 1, 1, False, []
    #)