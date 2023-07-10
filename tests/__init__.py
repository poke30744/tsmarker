from pathlib import Path

# sample files
samplesDir = Path(r"C:\Samples")
salor_moon_C_02_ts = samplesDir / "2021年01月05日19時30分00秒-美少女戦士セーラームーンCrystal #2_CS1(330)-1.ts"
salor_moon_C_02_ptsmap = samplesDir / "2021年01月05日19時30分00秒-美少女戦士セーラームーンCrystal #2_CS1(330)-1.ptsmap"
conan_ts = samplesDir / "2020年05月23日18時00分00秒-名探偵コナン「小五郎はBARにいる(前編)」[解][字][デ]_HD-1.ts"
conan_ptsmap = samplesDir / "_metadata" / "2020年05月23日18時00分00秒-名探偵コナン「小五郎はBARにいる(前編)」[解][字][デ]_HD-1.ptsmap"
conan_markermap = samplesDir / "_metadata" / "2020年05月23日18時00分00秒-名探偵コナン「小五郎はBARにいる(前編)」[解][字][デ]_HD-1.markermap"
conan_metadata = Path(r'\\acepc-gk3\Seagate 8T\categorized_mp4\アニメ - HD\名探偵コナン\_metadata')
conan_dataset = samplesDir / 'conan_dataset.csv'
jyounetsu_metadata = Path(r'\\acepc-gk3\Seagate 8T\categorized_mp4\ドキュメンタリー・教養\情熱大陸')
jyounetsu_dataset = samplesDir / 'jyounetsu_dataset.csv'
invalid_ts = samplesDir / "invalid.ts"
not_existing_ts = samplesDir / "not_existing.ts"
edges_with_boarder1 = samplesDir / "2021年03月01日19時00分00秒-美少女戦士セーラームーンS #25_trimmed_logo.edge.png"
edges_with_boarder2 = samplesDir / "2021年03月01日19時30分00秒-ルパン三世 PART4 #2「偽りのファンタジスタ」_trimmed_logo.edge.png"