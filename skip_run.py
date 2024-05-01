def print_model(file_name):
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('Column') or line.startswith('Average'):
                print('\033[1m' + '\033[94m' + line.strip() + '\033[0m')
            elif line.startswith('Validation') or line.startswith('Test'):
                print('\033[1m' + line.strip() + '\033[0m')
            else:
                print(line.strip())
    print()
