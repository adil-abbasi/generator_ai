from collections import deque
from copy import deepcopy

from collections import deque
from copy import deepcopy

import pandas as pd

from config import (
    RAW_FEATURES,
    LAG_FEATURES,
    ROLLING_FEATURES,
    DIFF_FEATURES,
)


class FeatureBuilder:
    """
    Maintains the last telemetry packets.

    Later this class will generate:
        - Lag Features
        - Rolling Features
        - Difference Features
        - Time Features
        - Derived Features
    """

    def __init__(self, history_size=4):

        self.history = deque(maxlen=history_size)

    # --------------------------------------------------

    def add_packet(self, packet: dict):
        """
        Store latest telemetry packet.
        """

        self.history.append(deepcopy(packet))

    # --------------------------------------------------

    def get_history(self):

        return list(self.history)

    # --------------------------------------------------

    def clear_history(self):

        self.history.clear()

    # --------------------------------------------------

    def history_size(self):

        return len(self.history)

    # --------------------------------------------------

    def latest_packet(self):

        if len(self.history) == 0:
            return None

        return self.history[-1]

    # --------------------------------------------------

    def build_features(self):

        if len(self.history) == 0:
            return None

        current = deepcopy(self.history[-1])

        features = {}

        # =====================================================
        # Raw Features
        # =====================================================

        for feature in RAW_FEATURES:

            features[feature] = current.get(feature)

        # =====================================================
        # Lag Features
        # =====================================================

        for feature in LAG_FEATURES:

            current_value = current.get(feature)

            # ---------- Lag 1 ----------

            if len(self.history) >= 2:

                lag1 = self.history[-2].get(feature)

            else:

                lag1 = current_value

            # ---------- Lag 2 ----------

            if len(self.history) >= 3:

                lag2 = self.history[-3].get(feature)

            else:

                lag2 = lag1

            # ---------- Lag 3 ----------

            if len(self.history) >= 4:

                lag3 = self.history[-4].get(feature)

            else:

                lag3 = lag2

            features[f"{feature}_lag1"] = lag1

            features[f"{feature}_lag2"] = lag2

            features[f"{feature}_lag3"] = lag3

        # =====================================================
        # Keep Remaining Fields
        # =====================================================

        for key, value in current.items():

            if key not in features:

                features[key] = value

        return features