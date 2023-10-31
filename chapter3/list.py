color1 = ['red', 'blue', 'green']
color2 = ['orange', 'black', 'white']

print(color1 + color2)  # 리스트 합치기

print(len(color1))  # 리스트 길이

total_color = color1 + color2

print(total_color)

print(color1 * 2)   # color1 리스트 2회 반복

print('blue' in color2) # color2 변수에서 문자열 'blue'의 존재 여부 반환

color3 = ['red', 'blue', 'green']
color3.append('white')   # 리스트에 'white' 추가
print(color3)

color4 = ['red', 'blue', 'green']
color4.extend(['black', 'purple'])  # 리스트에 새로운 리스트 추가
print(color4)

color5 = ['red', 'blue', 'green']
color5.insert(0, 'orange')  # 리스트의 특정 위치에 값 추가 (0번째 인덱스)
print(color5)

color6 = ['red', 'blue', 'green']
color6.remove('red')    # 리스트에 있는 특정 값 삭제
print(color6)

color7 = ['red', 'blue', 'green']
color7[0] = 'orange'    # 0번쨰 인덱스를 'orange'로 변경
print(color7)
del color7[0]   # 0번째 인덱스 값 삭제
print(color7)