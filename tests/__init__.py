from pathlib import Path

# sample files
samplesDir = Path(r"C:\repos\logoNet\Samples")
salor_moon_C_02_ts = samplesDir / "2021年01月05日19時30分00秒-美少女戦士セーラームーンCrystal #2_CS1(330)-1.ts"
salor_moon_C_02_ptsmap = samplesDir / "2021年01月05日19時30分00秒-美少女戦士セーラームーンCrystal #2_CS1(330)-1.ptsmap"
conan_ts = samplesDir / "2020年05月23日18時00分00秒-名探偵コナン「小五郎はBARにいる(前編)」[解][字][デ]_HD-1.ts"
conan_ptsmap = samplesDir / "2020年05月23日18時00分00秒-名探偵コナン「小五郎はBARにいる(前編)」[解][字][デ]_HD-1.ptsmap"
conan_markermap = samplesDir / "2020年05月23日18時00分00秒-名探偵コナン「小五郎はBARにいる(前編)」[解][字][デ]_HD-1.markermap"
conan_metadata = Path('\\\\raspberrypi4\\BUFFALO_Exp\\categorized_no_cm\\名探偵コナン\\_metadata')
conan_dataset = samplesDir / 'conan_dataset.csv'
invalid_ts = samplesDir / "invalid.ts"
not_existing_ts = samplesDir / "not_existing.ts"