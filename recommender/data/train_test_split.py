
class TrainTestSplit:
    def __init__(self, test_size, ts=False):
        """
        Params
        ------
        test_size : float
            Raio of test size. (0 <= test_size <= 1)

        ts : str
            Column name for timestamp
        """
        self.test_size = test_size
        self.ts = ts

    def split(self, df):
        """
        Stratified sampling by user_id and timestamp (if provided)

        Params
        ------
        df : pd.DataFrame
            Interaction information data frame.
            Column should be user_id | item_id | interaction | timestamp (if provided)
            For explicit feedback, interaction value will be something such as movie ratings.
            For implicit feedback, interaction value will be something such as number of clicks or purchases.
        """
        user2count = df["user_id"].value_counts().to_dict()
        sort_columns = ["user_id", "timestamp"] if self.ts else ["user_id"]
        df = df.sort_values(by=sort_columns)

        if self.ts:
            df = df.drop("timestamp", axis=1)

        count = 0
        before_user_id = df["user_id"].tolist()[0]
        test = []
        for user_id in df["user_id"]:
            # if a user has only one interaction data, just keep it to training data
            if user2count[user_id] == 1:
                test.append(0)
                before_user_id = user_id
                continue

            if user_id == before_user_id:
                count += 1
            else:
                count = 1
                before_user_id = user_id

            if count / user2count[user_id] < 1-self.test_size:
                test.append(0)
            else:
                test.append(1)
        train_idx = [i == 0 for i in test]
        val_idx = [i == 1 for i in test]
        return df.iloc[train_idx], df.iloc[val_idx]