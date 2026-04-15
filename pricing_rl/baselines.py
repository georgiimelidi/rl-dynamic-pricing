class FixedPricePolicy:
    def __init__(self, action: int):
        self.action = action

    def select_action(self, state):
        return self.action


class DecreasingPricePolicy:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def select_action(self, state):
        _, time_left = state[:2]

        if time_left > 0.66:
            return self.num_actions - 1
        if time_left > 0.33:
            return self.num_actions // 2
        return 0


class InventoryAwarePolicy:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def select_action(self, state):
        inventory_ratio, time_left = state[:2]

        if time_left > 0.75:
            return self.num_actions - 1
        if inventory_ratio > 0.6 and time_left < 0.4:
            return 0
        if inventory_ratio > 0.3 and time_left < 0.7:
            return self.num_actions // 2
        return self.num_actions - 1