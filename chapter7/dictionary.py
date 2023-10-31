country_code = {}   # 딕셔너리 생성
country_code = {"America": 1, "Korea": 82, "China": 86, "Japan": 81}
print(country_code)

print(country_code.keys())  # 딕셔너리의 키만 출력

country_code["German"] = 49 # 딕셔너리 추가
print(country_code)

print(country_code.values())    # 딕셔너리의 값만 출력

print(country_code.items()) # 딕셔너리의 데이터 출력

for k, v in country_code.items():
    print("Key:", k)
    print("Value:", v)

print("Korea" in country_code.keys())   # 키에 "Korea"가 있는지 확인

print(82 in country_code.values())  # 값에 82가 있는지 확인