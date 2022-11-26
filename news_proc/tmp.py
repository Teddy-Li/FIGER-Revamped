
ofp = open('../news_data/newscrawl_gparser_typing_input.json', 'w', encoding='utf-8')

for i in range(30):
    print(f"Processing section {i}")
    ifp = open(f'../news_data/newscrawl_gparser_typing_input_{i}.json', 'r', encoding='utf-8')
    for lidx, line in enumerate(ifp):
        if lidx % 100000 == 0:
            print(f"lidx: {lidx}")
        assert line[-1] == '\n'
        ofp.write(line)
    ifp.close()

print('Done!')
ofp.close()