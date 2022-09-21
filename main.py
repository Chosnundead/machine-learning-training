import modules.machinelearning as ml

task1 = ml.Division_By_Two_Category(
    {
        "x": [
            [10, 50],
            [20, 30],
            [25, 30],
            [20, 60],
            [15, 70],
            [40, 40],
            [30, 45],
            [20, 45],
            [40, 30],
            [7, 35],
        ],
        "y": [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
    }
)
task1.learning()
task1.show_graphic()
print(task1.check_result([[10, 40], [40, 1]]))

task2 = ml.Division_By_Three_Category(
    {
        "x": [
            [10, 50],
            [20, 30],
            [25, 30],
            [20, 60],
            [15, 70],
            [40, 40],
            [30, 45],
            [20, 45],
            [40, 30],
            [7, 35],
        ],
        "y": [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
    }
)
task2.learning()
task2.show_graphic()
