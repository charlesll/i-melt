import imelt
import numpy as np
import pandas as pd

######################################
## HELPER FUNCTIONS FOR PREDICTIONS ##
######################################


def generate_query_single(
    sio2=100.0,
    al2o3=0.0,
    na2o=0.0,
    k2o=0.0,
    mgo=0.0,
    cao=0.0,
    composition_mole=True,
):
    """Generates a query DataFrame for a single magma composition

    Args:
        sio2, al2o3, ..., cao: Oxide weight percentages (if composition_mole=False) or
            mole percentages (if composition_mole=True). Defaults to pure SiO2.
        composition_mole: Boolean indicating if input composition is in mole (True)
            or weight percent (False). Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the query with columns for T and oxide compositions.
    """

    db = pd.DataFrame(
        {
            "T": [
                0,
            ]
        }
    )

    for oxide in imelt.list_oxides():
        db[oxide] = locals()[oxide]  # Elegant way to set values from function arguments

    if db.loc[0, imelt.list_oxides()].sum() != 100.0:
        print("Warning: Composition does not sum to 100%. Renormalizing...")

    # convert in fractions and normalise.
    db = imelt.chimie_control(db).copy()

    if not composition_mole:
        print("Converting weight percent composition to mole fraction...")
        db = imelt.wt_mol(db)

    # add descriptors
    db = imelt.descriptors(db.loc[:, imelt.list_oxides()])

    return db.to_numpy()


def generate_query_range(
    oxide_ranges: dict,  # Use a dictionary to specify oxide ranges
    composition_mole=True,
    nb_values=10,
):
    """Generates a query DataFrame for multiple magma compositions within specified ranges

    Args:
        oxide_ranges: A dictionary specifying the initial and final values for each oxide.
            Keys should be oxide names (e.g., 'SiO2', 'TiO2'), and values should be lists or tuples
            of length 2: [min_value, max_value].
        composition_mole: Boolean indicating if input composition is in mole fraction (True)
            or weight percent (False). Defaults to True.
        nb_values: Number of values to generate within each oxide range (and for T and P).

    Returns:
        pd.DataFrame: A DataFrame containing the query with columns for
                      oxide compositions and descriptors.
    """

    if nb_values <= 1:
        raise ValueError(
            "nb_values must be greater than 1 to generate a range of values."
        )

    # Check if oxide_ranges dictionary contains the required oxides
    for oxide in imelt.list_oxides():
        if oxide not in oxide_ranges:
            raise ValueError(f"Missing oxide range: {oxide}")

    # Create all combinations of oxide values
    oxide_values = [
        np.linspace(oxide_ranges[oxide][0], oxide_ranges[oxide][1], nb_values)
        for oxide in imelt.list_oxides()
    ]

    db = pd.DataFrame(np.array(oxide_values).T, columns=imelt.list_oxides())

    if db.loc[0, imelt.list_oxides()].sum() != 100.0:
        print("Warning: Composition does not sum to 100%. Renormalizing...")

    # convert in fractions and normalise.
    db = imelt.chimie_control(db).copy()

    if not composition_mole:
        print("Converting weight percent composition to mole fraction...")
        db = imelt.wt_mol(db)

    # add descriptors
    db = imelt.descriptors(db.loc[:, imelt.list_oxides()])

    return db
