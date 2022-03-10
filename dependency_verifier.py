import pandas as pd


def attributes_by_activities(dataframe):
    activities = {}

    for activity in list(dataframe['activity']):
        attributes = []
        activity_frame = dataframe[dataframe['activity'] == activity]
        for attribute in activity_frame.columns.values:
            for idx in activity_frame.index.values:
                if not pd.isnull(activity_frame.at[idx, attribute]):
                    attributes.append(attribute)

        # if len(attributes)>0:
        # activities[activity] = sattributes)

    return activities


def get_rows_idx(case, activity1, activity2):
    # get the 2 rows where timestamp difference is minimal and timestamp1 < timestamp2
    rows_idx = []
    case_len= len(case.index)
    idx = 0
    while idx < case_len:
        if case.at[case.index[idx],'activity'] == activity1:
            row1_idx = idx
            idx += 1
            while idx < case_len:
                if case.at[case.index[idx],'activity'] == activity2:
                    row2_idx = idx
                    rows_idx.append((row1_idx, row2_idx))
                    idx += 1
                    break
                elif case.at[case.index[idx],'activity'] == activity1:
                    row1_idx = idx
                    idx += 1
                else:
                    idx += 1
        else:
            idx += 1

    return rows_idx


def compare_rows(row1, row2):
    attributes = []
    for attribute in row1.columns.values:
        if row1.at[row1.index[0], attribute] != row2.at[row2.index[0], attribute]:
            attributes.append(attribute)

    return attributes


def verify_dependency(path_log_csv, activity1, activity2):
    # log has to be csv not xes
    # note that activity1 occurs before activity2
    with open(path_log_csv, 'rb') as file:
        log = pd.read_csv(file, sep=';')

    case_ids = list(set(list(log['case_id'])))
    cases = log.groupby('case_id')

    for case_key, case_values in cases:
        # case = cases.get_group(case_key)
        case_values.sort_values(['timestamp'])
        rows_idx = get_rows_idx(case_values, activity1, activity2)

        attributes = []
        for row1_idx, row2_idx in rows_idx:
            row1 = case_values.loc[case_values.index == case_values.index[row1_idx]]
            row2 = case_values.loc[case_values.index == case_values.index[row2_idx]]
            row_attributes = compare_rows(row1, row2)
            attributes = (set(attributes).intersection(row_attributes))

    return attributes


def main():
    result = verify_dependency('./testing/test_workout_log.CSV', 'Entered Drink Refill Area', 'Left Drink Refill Area')
    print(result)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()