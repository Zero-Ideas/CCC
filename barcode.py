import re
barcodeAmount = int(input())
for _ in range(barcodeAmount):
    barcode = input().strip()
    uppercaseLetters = "".join([char for char in barcode if char.isupper()])
    numbers = re.findall(r'-?\d+', barcode) 
    sumOfNumbers = sum(map(int, numbers))
    print(uppercaseLetters + str(sumOfNumbers))
