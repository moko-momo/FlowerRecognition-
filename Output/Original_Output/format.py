import os

file_names = os.listdir("./")

for file_name in file_names:
    if file_name[0:3] != "for":
        f = open(file_name)
        contentList = f.read().split("\n")
        new_str_train = ""
        new_str_evaluate = ""
        train_count = 1
        evaluation_count = 1
        for line in contentList:
            if line[0:9] == "3458/3458":
                new_str_train += str("Train %d: " % train_count)
                train_count += 1
                new_str_train += line
                new_str_train += "\n"
            if line[0:7] == "865/865":
                new_str_evaluate += str("Evaluation %d: " % evaluation_count)
                evaluation_count += 1
                new_str_evaluate += line
                new_str_evaluate += "\n"
        f.close()
        new_filename = str("Formatted-" + file_name)
        new_f = open(str("../Formatted_Output/" + new_filename), "w")
        new_f.write(new_str_train + "\n\n" + new_str_evaluate)
        new_f.close()


