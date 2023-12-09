
from pipeline_functions import *



def main():
    # Create synthetic dataset with 2 longitudinal views
    data, y = synthetic_data(eps=0.75, eta=0.75, plots=True)

    # Perform variable selection using method = 'LMM', 'DGB', 'JPTA' or 'nothing'
    data = variable_selection(data, y, method='LMM', top_features='default')

    # Performs feature extraction using methods 'EC', 'FPCA' or 'nothing'
    results = featureExtraction_DeepIDA_GRU(data, y, list_of_conversions=['EC', 'nothing'], n_epochs=10)

    # Print results
    print('Train accuracy:', results[0])  # Train accuracy, test accuracy
    print('Test accuracy:', results[1])

    return


if __name__ == "__main__":
    main()