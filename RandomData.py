import random


class HumanRandomData:
    """
    Random data gets size of data as parameter 1
    and a boolean as a parameter 2 to get extra output in console during generating.
    data = [[age, weight, height]]
    targets = [correct_output]
    correct_output is a variable contains 0 or 1
    where 0 (false) is not a human and 1 (true) is a human
    """

    training = False
    debug = False

    def __init__(self, data_size, debug=False):
        self.data_size = data_size
        self.debug = debug

        self.data = []
        self.targets = []

        self.generate_data()
        if self.debug:
            self.__str__()

    def generate_data(self):

        count_humans = 0
        count_others = 0

        # Random pick between humans and others
        for x in range(self.data_size):

            # Humans
            if random.randint(0, 1) == 1:
                count_humans += 1
                data_row = self.make_human_data()
                self.targets.append(data_row[1])
                self.data.append(data_row[0])

            # Others
            else:
                count_others += 1
                data_row = self.make_other_data()
                while self.check_if_human(data_row[0]):
                    data_row = self.make_other_data()
                self.targets.append(data_row[1])
                self.data.append(data_row[0])


        self.print_string_with_star_lines("### --- Data generated --- ###")
        if self.debug:
            print("Humans: " + str(count_humans))
            print("Others: " + str(count_others))
            print("Training data: ", self.training)
            print()

    def check_if_human(self, data_row):
        age = data_row[0]
        weight = data_row[1]
        height = data_row[2]

        if age > 90:
            return False
        elif 2 <= age <= 5 and 10 <= weight <= 15 and 90 <= height <= 120:
            return True
        elif 10 >= age >= 5 and 15 <= weight <= 50 and 121 <= height <= 150:
            return True
        elif 16 > age >= 10 and 50 <= weight <= 85 and 163 <= height <= 182:
            return True
        elif 21 > age >= 16 and 60 <= weight <= 90 and 170 <= height <= 190:
            return True
        elif 70 > age >= 21 and 65 <= weight <= 95 and 175 <= height <= 195:
            return True
        elif age >= 70 and 55 <= weight <= 80 and 173 <= height <= 185:
            return True

        return False

    def make_human_data(self):

        age = random.randint(2, 90)

        if age <= 5:
            weight = random.randint(10, 15)
            height = random.randint(90, 120)
        elif 10 > age > 5:
            weight = random.randint(15, 50)
            height = random.randint(121, 150)
        elif 16 > age >= 10:
            weight = random.randint(50, 85)
            height = random.randint(163, 182)
        elif 21 > age >= 16:
            weight = random.randint(60, 90)
            height = random.randint(170, 190)
        elif 70 > age >= 21:
            weight = random.randint(65, 95)
            height = random.randint(175, 195)
        else:
            weight = random.randint(55, 80)
            height = random.randint(173, 185)

        target = 1

        return [[float(age), float(weight), float(height)], float(target)]

    def make_other_data(self):

        weight = random.randint(10, 120)
        height = random.randint(50, 210)

        age = random.randint(2, 90)

        target = 0

        return [[float(age), float(weight), float(height)], float(target)]

    def __str__(self):
        for i, element in enumerate(self.data):
            print(element, self.targets[i])
        print()


    def print_string_with_star_lines(self, text=None):
        if text is None:
            print("\n**********************************************\n")
        else:
            print("*"*len(text))
            print(text)
            print("*"*len(text))
            print()

    @staticmethod
    def normalize_data(input_data):
        for data_row in input_data:

            for i, data_element in enumerate(data_row):

                # data element 0 = Age;
                # Max value is estimated to be 90
                if i == 0:
                    data_row[i] = round((data_row[i]/90), 4)

                # data element 1 = Weight
                # Max value is estimated to be 150
                elif i == 1:
                    data_row[i] = round((data_row[i]/150), 4)

                # data element 2 = Height
                # Max value is estimated to be 210
                else:
                    data_row[i] = round((data_row[i]/210), 4)
        return input_data
