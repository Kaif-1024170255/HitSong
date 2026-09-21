# Data Audit Report

Audit of all raw CSV files in `data/raw/` conducted following Rule 1 (never fabricate a number; print, read, then write).

## File: `billboard_24years_lyrics_spotify.csv`

- **Filename:** `billboard_24years_lyrics_spotify.csv`
- **Rows:** 3,397
- **Columns:** 26

### Full Column List (dtype, non-null count, % missing)

| Column           | Dtype   | Non-Null Count | Null Count | % Missing |
| ---------------- | ------- | -------------- | ---------- | --------- |
| ranking          | int64   | 3,397          | 0          | 0.00%     |
| song             | str     | 3,397          | 0          | 0.00%     |
| band_singer      | str     | 3,397          | 0          | 0.00%     |
| songurl          | str     | 3,397          | 0          | 0.00%     |
| titletext        | str     | 3,397          | 0          | 0.00%     |
| url              | str     | 3,397          | 0          | 0.00%     |
| year             | int64   | 3,397          | 0          | 0.00%     |
| lyrics           | str     | 3,397          | 0          | 0.00%     |
| uri              | str     | 3,397          | 0          | 0.00%     |
| danceability     | float64 | 486            | 2,911      | 85.69%    |
| energy           | float64 | 486            | 2,911      | 85.69%    |
| key              | float64 | 486            | 2,911      | 85.69%    |
| loudness         | float64 | 486            | 2,911      | 85.69%    |
| mode             | float64 | 486            | 2,911      | 85.69%    |
| speechiness      | float64 | 486            | 2,911      | 85.69%    |
| acousticness     | float64 | 486            | 2,911      | 85.69%    |
| instrumentalness | float64 | 486            | 2,911      | 85.69%    |
| liveness         | float64 | 486            | 2,911      | 85.69%    |
| valence          | float64 | 486            | 2,911      | 85.69%    |
| tempo            | float64 | 486            | 2,911      | 85.69%    |
| type             | str     | 486            | 2,911      | 85.69%    |
| id               | str     | 486            | 2,911      | 85.69%    |
| track_href       | str     | 486            | 2,911      | 85.69%    |
| analysis_url     | str     | 486            | 2,911      | 85.69%    |
| duration_ms      | float64 | 486            | 2,911      | 85.69%    |
| time_signature   | float64 | 486            | 2,911      | 85.69%    |

### `df.head(5)`

| ranking | song        | band_singer     | songurl                         | titletext   | url                         | year | lyrics                                   | uri                                  | danceability | energy | key | loudness | mode | speechiness | acousticness | instrumentalness       | liveness | valence | tempo   | type           | id                     | track_href                               | analysis_url                             | duration_ms | time_signature |
| ------- | ----------- | --------------- | ------------------------------- | ----------- | --------------------------- | ---- | ---------------------------------------- | ------------------------------------ | ------------ | ------ | --- | -------- | ---- | ----------- | ------------ | ---------------------- | -------- | ------- | ------- | -------------- | ---------------------- | ---------------------------------------- | ---------------------------------------- | ----------- | -------------- |
| 1       | Breathe     | Faith Hill      | /wiki/Breathe_(Faith_Hill_song) | Breathe     | /wiki/Faith_Hill            | 2000 | I can feel the magic floating in the ... | spotify:track:3y4LxiYMgDl4RethdzpmNe | 0.529        | 0.496  | 7.0 | -9.007   | 1.0  | 0.029       | 0.173        | 0.0                    | 0.251    | 0.278   | 136.859 | audio_features | 3y4LxiYMgDl4RethdzpmNe | https://api.spotify.com/v1/tracks/3y4... | https://api.spotify.com/v1/audio-anal... | 250547.0    | 4.0            |
| 2       | Smooth      | Santana         | /wiki/Smooth_(Santana_song)     | Smooth      | /wiki/Santana_(band)        | 2000 | Man, it's a hot one Like seven inches... | spotify:track:0n2SEXB2qoRQg171q7XqeW | 0.609        | 0.923  | 9.0 | -3.908   | 1.0  | 0.0338      | 0.16         | 4.7300000000000005e-06 | 0.295    | 0.961   | 115.996 | audio_features | 0n2SEXB2qoRQg171q7XqeW | https://api.spotify.com/v1/tracks/0n2... | https://api.spotify.com/v1/audio-anal... | 294987.0    | 4.0            |
| 2       | Smooth      | Rob Thomas      | /wiki/Smooth_(Santana_song)     | Smooth      | /wiki/Rob_Thomas_(musician) | 2000 | Man, it's a hot one Like seven inches... | spotify:track:5IALWUYK0zDSEmZgb4ICvc | 0.59         | 0.637  | 9.0 | -9.171   | 1.0  | 0.0301      | 0.00225      | 0.807                  | 0.299    | 0.724   | 115.983 | audio_features | 5IALWUYK0zDSEmZgb4ICvc | https://api.spotify.com/v1/tracks/5IA... | https://api.spotify.com/v1/audio-anal... | 244924.0    | 4.0            |
| 3       | Maria Maria | Santana         | /wiki/Maria_Maria               | Maria Maria | /wiki/Santana_(band)        | 2000 | Ladies and gents, turn up your sound ... | spotify:track:3XKIUb7HzIF1Vu9usunMzc | 0.777        | 0.601  | 2.0 | -5.931   | 1.0  | 0.126       | 0.0406       | 0.00201                | 0.0348   | 0.68    | 97.911  | audio_features | 3XKIUb7HzIF1Vu9usunMzc | https://api.spotify.com/v1/tracks/3XK... | https://api.spotify.com/v1/audio-anal... | 261973.0    | 4.0            |
| 3       | Maria Maria | The Product G&B | /wiki/Maria_Maria               | Maria Maria | /wiki/The_Product_G%26B     | 2000 | Turn up this sound system To the soun... | spotify:track:3XKIUb7HzIF1Vu9usunMzc |              |        |     |          |      |             |              |                        |          |         |         |                |                        |                                          |                                          |             |                |

### Numeric Columns Summary (Min, Max, Mean, Median, % Missing)

| Column           | Min        | Max         | Mean        | Median      | % Missing |
| ---------------- | ---------- | ----------- | ----------- | ----------- | --------- |
| ranking          | 1.0000     | 100.0000    | 50.1021     | 50.0000     | 0.00%     |
| year             | 2000.0000  | 2023.0000   | 2011.7692   | 2012.0000   | 0.00%     |
| danceability     | 0.1900     | 0.9560      | 0.6675      | 0.6770      | 85.69%    |
| energy           | 0.1190     | 0.9800      | 0.6852      | 0.6950      | 85.69%    |
| key              | 0.0000     | 11.0000     | 5.0885      | 5.0000      | 85.69%    |
| loudness         | -17.6920   | -1.2990     | -6.2254     | -5.8660     | 85.69%    |
| mode             | 0.0000     | 1.0000      | 0.5926      | 1.0000      | 85.69%    |
| speechiness      | 0.0236     | 0.5050      | 0.1122      | 0.0583      | 85.69%    |
| acousticness     | 0.0001     | 0.9550      | 0.1635      | 0.0849      | 85.69%    |
| instrumentalness | 0.0000     | 0.9250      | 0.0189      | 0.0000      | 85.69%    |
| liveness         | 0.0272     | 0.9570      | 0.1700      | 0.1130      | 85.69%    |
| valence          | 0.0612     | 0.9690      | 0.5997      | 0.6415      | 85.69%    |
| tempo            | 62.8760    | 203.7460    | 116.6092    | 106.1530    | 85.69%    |
| duration_ms      | 74080.0000 | 432147.0000 | 248755.3169 | 242753.0000 | 85.69%    |
| time_signature   | 1.0000     | 5.0000      | 3.9774      | 4.0000      | 85.69%    |

### Categorical / Object Columns Summary

#### Column: `song`
- **Unique values:** 2,080
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value             | Count | % of Total |
| ----------------- | ----- | ---------- |
| I Like It         | 9     | 0.26%      |
| Beautiful         | 9     | 0.26%      |
| Forever           | 9     | 0.26%      |
| Stay              | 8     | 0.24%      |
| Rockstar          | 7     | 0.21%      |
| Holidae In        | 6     | 0.18%      |
| Like a G6         | 6     | 0.18%      |
| Say Something     | 6     | 0.18%      |
| Party Rock Anthem | 6     | 0.18%      |
| Mercy             | 6     | 0.18%      |

#### Column: `band_singer`
- **Unique values:** 1,039
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value         | Count | % of Total |
| ------------- | ----- | ---------- |
| Drake         | 70    | 2.06%      |
| Rihanna       | 46    | 1.35%      |
| Chris Brown   | 36    | 1.06%      |
| Lil Wayne     | 35    | 1.03%      |
| Taylor Swift  | 34    | 1.00%      |
| Nicki Minaj   | 34    | 1.00%      |
| Justin Bieber | 32    | 0.94%      |
| Ludacris      | 29    | 0.85%      |
| Beyoncé       | 27    | 0.79%      |
| Usher         | 26    | 0.77%      |

#### Column: `songurl`
- **Unique values:** 2,172
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value                                                        | Count | % of Total |
| ------------------------------------------------------------ | ----- | ---------- |
| /wiki/Forever_(Drake_song)                                   | 8     | 0.24%      |
| /wiki/Holidae_In                                             | 6     | 0.18%      |
| /wiki/Like_a_G6                                              | 6     | 0.18%      |
| /wiki/Party_Rock_Anthem                                      | 6     | 0.18%      |
| /wiki/Blurred_Lines                                          | 6     | 0.18%      |
| /wiki/Bang_Bang_(Jessie_J,_Ariana_Grande_and_Nicki_Minaj_... | 6     | 0.18%      |
| /wiki/Lean_On                                                | 6     | 0.18%      |
| /wiki/Sucker_for_Pain                                        | 6     | 0.18%      |
| /wiki/Starving_(song)                                        | 6     | 0.18%      |
| /wiki/1-800-273-8255_(song)                                  | 6     | 0.18%      |

#### Column: `titletext`
- **Unique values:** 2,078
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value             | Count | % of Total |
| ----------------- | ----- | ---------- |
| I Like It         | 9     | 0.26%      |
| Beautiful         | 9     | 0.26%      |
| Forever           | 9     | 0.26%      |
| Stay              | 8     | 0.24%      |
| Rockstar          | 7     | 0.21%      |
| Holidae In        | 6     | 0.18%      |
| Like a G6         | 6     | 0.18%      |
| Say Something     | 6     | 0.18%      |
| Party Rock Anthem | 6     | 0.18%      |
| Mercy             | 6     | 0.18%      |

#### Column: `url`
- **Unique values:** 1,042
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value                  | Count | % of Total |
| ---------------------- | ----- | ---------- |
| /wiki/Drake_(musician) | 59    | 1.74%      |
| /wiki/Rihanna          | 46    | 1.35%      |
| /wiki/Lil_Wayne        | 35    | 1.03%      |
| /wiki/Taylor_Swift     | 34    | 1.00%      |
| /wiki/Nicki_Minaj      | 34    | 1.00%      |
| /wiki/Justin_Bieber    | 32    | 0.94%      |
| /wiki/Chris_Brown      | 31    | 0.91%      |
| /wiki/Ludacris         | 29    | 0.85%      |
| /wiki/Usher_(musician) | 26    | 0.77%      |
| /wiki/Ariana_Grande    | 26    | 0.77%      |

#### Column: `lyrics`
- **Unique values:** 2,330
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value                                                        | Count | % of Total |
| ------------------------------------------------------------ | ----- | ---------- |
| Bomb ass pussy Ma, ooh you got that bomb, know you got it... | 6     | 0.18%      |
| Poppin bottles in the ice, like a blizzard When we drink ... | 6     | 0.18%      |
| Party Rock Yeah Woo! Let's go!  Party rock is in the hous... | 6     | 0.18%      |
| She got a body like an hourglass But I can give it to you... | 6     | 0.18%      |
| I've been on the low, I been taking my time I feel like I... | 6     | 0.18%      |
| Yeah, baby, I like it like that You gotta believe me when... | 6     | 0.18%      |
| CashMoneyAP Kill 'em, get 'em, get 'em, oh Oh (I'm tryna ... | 6     | 0.18%      |
| I gotta feed the streets, my pistol gon' bleed the street... | 6     | 0.18%      |
| I will wait for you, for you Early in the mornin', late a... | 6     | 0.18%      |
| I went to the moped store, said "Fuck it" And salesman's ... | 5     | 0.15%      |

#### Column: `uri`
- **Unique values:** 2,319
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value                                | Count | % of Total |
| ------------------------------------ | ----- | ---------- |
| spotify:track:4DvhkX2ic4zWkQeWMwQ2qf | 6     | 0.18%      |
| spotify:track:0IkKz2J93C94Ei4BvDop7P | 6     | 0.18%      |
| spotify:track:0n4bITAu0Y0nigrz3MFJMb | 6     | 0.18%      |
| spotify:track:0puf9yIluy9W0vpMEUoAnN | 6     | 0.18%      |
| spotify:track:01aTsQoKoeXofSTvKuunzv | 6     | 0.18%      |
| spotify:track:4Ce37cRWvM1vIGGynKcs22 | 6     | 0.18%      |
| spotify:track:5tz69p7tJuGPeMGwNTxYuV | 6     | 0.18%      |
| spotify:track:58q2HKrzhC3ozto2nDdN4z | 6     | 0.18%      |
| spotify:track:7FGq80cy8juXBCD2nrqdWU | 6     | 0.18%      |
| spotify:track:0PvFJmanyNQMseIFrU708S | 6     | 0.18%      |

#### Column: `type`
- **Unique values:** 1
- **Missing count:** 2,911 (85.69%)
- **Top 10 most frequent values:**
| Value          | Count | % of Total |
| -------------- | ----- | ---------- |
| nan            | 2,911 | 85.69%     |
| audio_features | 486   | 14.31%     |

#### Column: `id`
- **Unique values:** 457
- **Missing count:** 2,911 (85.69%)
- **Top 10 most frequent values:**
| Value                  | Count | % of Total |
| ---------------------- | ----- | ---------- |
| nan                    | 2,911 | 85.69%     |
| 4NF3tP9JijuggkC67DY0v1 | 3     | 0.09%      |
| 6ZOBP3NvffbU4SZcrnt1k6 | 2     | 0.06%      |
| 0eKyHwckh9vQb8ncZ2DXCs | 2     | 0.06%      |
| 6vgN6KGstss0GnBbKlMNK9 | 2     | 0.06%      |
| 1ak0S3NhwWrUgNlQhJ1412 | 2     | 0.06%      |
| 65B1tEOv5W294uCKbmEcFV | 2     | 0.06%      |
| 1C2Q3AymKS17bw7T5gfq6p | 2     | 0.06%      |
| 3aw9iWUQ3VrPQltgwvN9Xu | 2     | 0.06%      |
| 0Uyh92tLyb9JawG8lmWCzJ | 2     | 0.06%      |

#### Column: `track_href`
- **Unique values:** 457
- **Missing count:** 2,911 (85.69%)
- **Top 10 most frequent values:**
| Value                                                    | Count | % of Total |
| -------------------------------------------------------- | ----- | ---------- |
| nan                                                      | 2,911 | 85.69%     |
| https://api.spotify.com/v1/tracks/4NF3tP9JijuggkC67DY0v1 | 3     | 0.09%      |
| https://api.spotify.com/v1/tracks/6ZOBP3NvffbU4SZcrnt1k6 | 2     | 0.06%      |
| https://api.spotify.com/v1/tracks/0eKyHwckh9vQb8ncZ2DXCs | 2     | 0.06%      |
| https://api.spotify.com/v1/tracks/6vgN6KGstss0GnBbKlMNK9 | 2     | 0.06%      |
| https://api.spotify.com/v1/tracks/1ak0S3NhwWrUgNlQhJ1412 | 2     | 0.06%      |
| https://api.spotify.com/v1/tracks/65B1tEOv5W294uCKbmEcFV | 2     | 0.06%      |
| https://api.spotify.com/v1/tracks/1C2Q3AymKS17bw7T5gfq6p | 2     | 0.06%      |
| https://api.spotify.com/v1/tracks/3aw9iWUQ3VrPQltgwvN9Xu | 2     | 0.06%      |
| https://api.spotify.com/v1/tracks/0Uyh92tLyb9JawG8lmWCzJ | 2     | 0.06%      |

#### Column: `analysis_url`
- **Unique values:** 457
- **Missing count:** 2,911 (85.69%)
- **Top 10 most frequent values:**
| Value                                                        | Count | % of Total |
| ------------------------------------------------------------ | ----- | ---------- |
| nan                                                          | 2,911 | 85.69%     |
| https://api.spotify.com/v1/audio-analysis/4NF3tP9JijuggkC... | 3     | 0.09%      |
| https://api.spotify.com/v1/audio-analysis/6ZOBP3NvffbU4SZ... | 2     | 0.06%      |
| https://api.spotify.com/v1/audio-analysis/0eKyHwckh9vQb8n... | 2     | 0.06%      |
| https://api.spotify.com/v1/audio-analysis/6vgN6KGstss0GnB... | 2     | 0.06%      |
| https://api.spotify.com/v1/audio-analysis/1ak0S3NhwWrUgNl... | 2     | 0.06%      |
| https://api.spotify.com/v1/audio-analysis/65B1tEOv5W294uC... | 2     | 0.06%      |
| https://api.spotify.com/v1/audio-analysis/1C2Q3AymKS17bw7... | 2     | 0.06%      |
| https://api.spotify.com/v1/audio-analysis/3aw9iWUQ3VrPQlt... | 2     | 0.06%      |
| https://api.spotify.com/v1/audio-analysis/0Uyh92tLyb9JawG... | 2     | 0.06%      |

### Duplicate Analysis

- **Exact duplicate rows:** 0
- **Duplicates on (`song` + `band_singer`):** 320

---

## File: `dataset.csv`

- **Filename:** `dataset.csv`
- **Rows:** 114,000
- **Columns:** 21

### Full Column List (dtype, non-null count, % missing)

| Column           | Dtype   | Non-Null Count | Null Count | % Missing |
| ---------------- | ------- | -------------- | ---------- | --------- |
| Unnamed: 0       | int64   | 114,000        | 0          | 0.00%     |
| track_id         | str     | 114,000        | 0          | 0.00%     |
| artists          | str     | 113,999        | 1          | 0.00%     |
| album_name       | str     | 113,999        | 1          | 0.00%     |
| track_name       | str     | 113,999        | 1          | 0.00%     |
| popularity       | int64   | 114,000        | 0          | 0.00%     |
| duration_ms      | int64   | 114,000        | 0          | 0.00%     |
| explicit         | bool    | 114,000        | 0          | 0.00%     |
| danceability     | float64 | 114,000        | 0          | 0.00%     |
| energy           | float64 | 114,000        | 0          | 0.00%     |
| key              | int64   | 114,000        | 0          | 0.00%     |
| loudness         | float64 | 114,000        | 0          | 0.00%     |
| mode             | int64   | 114,000        | 0          | 0.00%     |
| speechiness      | float64 | 114,000        | 0          | 0.00%     |
| acousticness     | float64 | 114,000        | 0          | 0.00%     |
| instrumentalness | float64 | 114,000        | 0          | 0.00%     |
| liveness         | float64 | 114,000        | 0          | 0.00%     |
| valence          | float64 | 114,000        | 0          | 0.00%     |
| tempo            | float64 | 114,000        | 0          | 0.00%     |
| time_signature   | int64   | 114,000        | 0          | 0.00%     |
| track_genre      | str     | 114,000        | 0          | 0.00%     |

### `df.head(5)`

| Unnamed: 0 | track_id               | artists                | album_name                               | track_name                 | popularity | duration_ms | explicit | danceability | energy | key | loudness | mode | speechiness | acousticness | instrumentalness | liveness | valence | tempo   | time_signature | track_genre |
| ---------- | ---------------------- | ---------------------- | ---------------------------------------- | -------------------------- | ---------- | ----------- | -------- | ------------ | ------ | --- | -------- | ---- | ----------- | ------------ | ---------------- | -------- | ------- | ------- | -------------- | ----------- |
| 0          | 5SuOikwiRyPMVoIQDJUgSV | Gen Hoshino            | Comedy                                   | Comedy                     | 73         | 230666      | False    | 0.676        | 0.461  | 1   | -6.746   | 0    | 0.143       | 0.0322       | 1.01e-06         | 0.358    | 0.715   | 87.917  | 4              | acoustic    |
| 1          | 4qPNDBW1i3p13qLCt0Ki3A | Ben Woodward           | Ghost (Acoustic)                         | Ghost - Acoustic           | 55         | 149610      | False    | 0.42         | 0.166  | 1   | -17.235  | 1    | 0.0763      | 0.924        | 5.56e-06         | 0.101    | 0.267   | 77.489  | 4              | acoustic    |
| 2          | 1iJBSr7s7jYXzM8EGcbK5b | Ingrid Michaelson;ZAYN | To Begin Again                           | To Begin Again             | 57         | 210826      | False    | 0.438        | 0.359  | 0   | -9.734   | 1    | 0.0557      | 0.21         | 0.0              | 0.117    | 0.12    | 76.332  | 4              | acoustic    |
| 3          | 6lfxq3CG4xtTiEg7opyCyx | Kina Grannis           | Crazy Rich Asians (Original Motion Pi... | Can't Help Falling In Love | 71         | 201933      | False    | 0.266        | 0.0596 | 0   | -18.515  | 1    | 0.0363      | 0.905        | 7.07e-05         | 0.132    | 0.143   | 181.74  | 3              | acoustic    |
| 4          | 5vjLSffimiIP26QG5WcN2K | Chord Overstreet       | Hold On                                  | Hold On                    | 82         | 198853      | False    | 0.618        | 0.443  | 2   | -9.681   | 1    | 0.0526      | 0.469        | 0.0              | 0.0829   | 0.167   | 119.949 | 4              | acoustic    |

### Numeric Columns Summary (Min, Max, Mean, Median, % Missing)

| Column           | Min      | Max          | Mean        | Median      | % Missing |
| ---------------- | -------- | ------------ | ----------- | ----------- | --------- |
| Unnamed: 0       | 0.0000   | 113999.0000  | 56999.5000  | 56999.5000  | 0.00%     |
| popularity       | 0.0000   | 100.0000     | 33.2385     | 35.0000     | 0.00%     |
| duration_ms      | 0.0000   | 5237295.0000 | 228029.1531 | 212906.0000 | 0.00%     |
| danceability     | 0.0000   | 0.9850       | 0.5668      | 0.5800      | 0.00%     |
| energy           | 0.0000   | 1.0000       | 0.6414      | 0.6850      | 0.00%     |
| key              | 0.0000   | 11.0000      | 5.3091      | 5.0000      | 0.00%     |
| loudness         | -49.5310 | 4.5320       | -8.2590     | -7.0040     | 0.00%     |
| mode             | 0.0000   | 1.0000       | 0.6376      | 1.0000      | 0.00%     |
| speechiness      | 0.0000   | 0.9650       | 0.0847      | 0.0489      | 0.00%     |
| acousticness     | 0.0000   | 0.9960       | 0.3149      | 0.1690      | 0.00%     |
| instrumentalness | 0.0000   | 1.0000       | 0.1560      | 0.0000      | 0.00%     |
| liveness         | 0.0000   | 1.0000       | 0.2136      | 0.1320      | 0.00%     |
| valence          | 0.0000   | 0.9950       | 0.4741      | 0.4640      | 0.00%     |
| tempo            | 0.0000   | 243.3720     | 122.1478    | 122.0170    | 0.00%     |
| time_signature   | 0.0000   | 5.0000       | 3.9040      | 4.0000      | 0.00%     |

### Categorical / Object Columns Summary

#### Column: `track_id`
- **Unique values:** 89,741
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value                  | Count | % of Total |
| ---------------------- | ----- | ---------- |
| 6S3JlDAGk3uu3NtZbPnuhS | 9     | 0.01%      |
| 2kkvB3RNRzwjFdGhaUA0tz | 8     | 0.01%      |
| 2Ey6v4Sekh3Z0RUSISRosD | 8     | 0.01%      |
| 2vU6bm5hVF2idVknGzqyPL | 7     | 0.01%      |
| 4GPQDyw9hC1DiZVh0ouDVL | 7     | 0.01%      |
| 5ftfVzSLIi5ZxYdNbRtf41 | 7     | 0.01%      |
| 6bzWr3EpSEolVwlbLk58il | 7     | 0.01%      |
| 2aaClnypAakdAmLw74JXxB | 7     | 0.01%      |
| 4WJTKbNJQ41zXnb84jSWaj | 7     | 0.01%      |
| 0e5LcankE0UyJUuCoq1uH2 | 7     | 0.01%      |

#### Column: `artists`
- **Unique values:** 31,437
- **Missing count:** 1 (0.00%)
- **Top 10 most frequent values:**
| Value           | Count | % of Total |
| --------------- | ----- | ---------- |
| The Beatles     | 279   | 0.24%      |
| George Jones    | 271   | 0.24%      |
| Stevie Wonder   | 236   | 0.21%      |
| Linkin Park     | 224   | 0.20%      |
| Ella Fitzgerald | 222   | 0.19%      |
| Prateek Kuhad   | 217   | 0.19%      |
| Feid            | 202   | 0.18%      |
| Chuck Berry     | 190   | 0.17%      |
| Håkan Hellström | 183   | 0.16%      |
| OneRepublic     | 181   | 0.16%      |

#### Column: `album_name`
- **Unique values:** 46,589
- **Missing count:** 1 (0.00%)
- **Top 10 most frequent values:**
| Value                       | Count | % of Total |
| --------------------------- | ----- | ---------- |
| Alternative Christmas 2022  | 195   | 0.17%      |
| Feliz Cumpleaños con Perreo | 184   | 0.16%      |
| Metal                       | 143   | 0.13%      |
| Halloween con perreito      | 123   | 0.11%      |
| Halloween Party 2022        | 115   | 0.10%      |
| The Complete Hank Williams  | 111   | 0.10%      |
| Fiesta portatil             | 110   | 0.10%      |
| Frescura y Perreo           | 106   | 0.09%      |
| Esto me suena a Farra       | 105   | 0.09%      |
| Perreo en Halloween         | 103   | 0.09%      |

#### Column: `track_name`
- **Unique values:** 73,608
- **Missing count:** 1 (0.00%)
- **Top 10 most frequent values:**
| Value                          | Count | % of Total |
| ------------------------------ | ----- | ---------- |
| Run Rudolph Run                | 151   | 0.13%      |
| Halloween                      | 88    | 0.08%      |
| Frosty The Snowman             | 81    | 0.07%      |
| Little Saint Nick - 1991 Remix | 76    | 0.07%      |
| Last Last                      | 75    | 0.07%      |
| Christmas Time                 | 72    | 0.06%      |
| CÓMO SE SIENTE - Remix         | 64    | 0.06%      |
| Sleigh Ride                    | 61    | 0.05%      |
| RUMBATÓN                       | 60    | 0.05%      |
| X ÚLTIMA VEZ                   | 58    | 0.05%      |

#### Column: `explicit`
- **Unique values:** 2
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value | Count   | % of Total |
| ----- | ------- | ---------- |
| False | 104,253 | 91.45%     |
| True  | 9,747   | 8.55%      |

#### Column: `track_genre`
- **Unique values:** 114
- **Missing count:** 0 (0.00%)
- **Top 10 most frequent values:**
| Value       | Count | % of Total |
| ----------- | ----- | ---------- |
| acoustic    | 1,000 | 0.88%      |
| afrobeat    | 1,000 | 0.88%      |
| alt-rock    | 1,000 | 0.88%      |
| alternative | 1,000 | 0.88%      |
| ambient     | 1,000 | 0.88%      |
| anime       | 1,000 | 0.88%      |
| black-metal | 1,000 | 0.88%      |
| bluegrass   | 1,000 | 0.88%      |
| blues       | 1,000 | 0.88%      |
| brazil      | 1,000 | 0.88%      |

### Duplicate Analysis

- **Exact duplicate rows:** 0
- **Duplicates on (`track_name` + `artists`):** 32,656
- **Duplicates excluding index column `Unnamed: 0`:** 450
- **Duplicates on `track_id`:** 24,259 (unique `track_id` count: 89,741)

---

## Explicit Questions & Answers

### 1. Which column is or could become the binary target `charted`?

**Answer: UNCERTAIN**

**Detailed Findings:**
- Neither dataset contains a column named `charted` or an explicit binary indicator of hit/non-hit status.
- In `billboard_24years_lyrics_spotify.csv`: Every row represents a song that charted on the Billboard Hot 100 Year-End charts (ranking 1 to 100) between 2000 and 2023. There are no negative (uncharted) examples in this dataset. Furthermore, Spotify audio feature columns (`danceability`, `energy`, etc.) are missing for 2,911 out of 3,397 rows (85.69% missing), leaving only 486 rows with audio features.
- In `dataset.csv`: This is a 114,000-track multi-genre Spotify dataset (114 genres, 1,000 tracks per genre) with complete audio features. However, it does not have a Billboard charting flag.
- Potential candidate approaches for constructing `charted`:
  1. Cross-referencing `dataset.csv` tracks against `billboard_24years_lyrics_spotify.csv` by track name and artist (or Spotify ID).
  2. Binarizing Spotify `popularity` in `dataset.csv` using a defined threshold (e.g. `popularity >= 50` or `popularity >= 70`).
- Because picking either approach makes a major assumption about the ground truth label, under Rule 6 this must not be chosen silently.

### 2. Which columns map to each required audio/metadata feature?

| Concept          | In billboard_24years_lyrics_spotify.csv | In dataset.csv                   |
| ---------------- | --------------------------------------- | -------------------------------- |
| tempo            | tempo                                   | tempo                            |
| energy           | energy                                  | energy                           |
| danceability     | danceability                            | danceability                     |
| valence          | valence                                 | valence                          |
| loudness         | loudness                                | loudness                         |
| acousticness     | acousticness                            | acousticness                     |
| instrumentalness | instrumentalness                        | instrumentalness                 |
| speechiness      | speechiness                             | speechiness                      |
| duration         | duration_ms                             | duration_ms                      |
| genre            | UNCERTAIN / None (column absent)        | track_genre                      |
| release year     | year (Billboard chart year, 2000-2023)  | UNCERTAIN / None (column absent) |

**Notes on Audio Features & Metadata:**
- For `tempo`, `energy`, `danceability`, `valence`, `loudness`, `acousticness`, `instrumentalness`, `speechiness`, and `duration`: identical column names exist in both datasets. However, in `billboard_24years_lyrics_spotify.csv`, these audio features have 85.69% missing values (only 486 non-null rows). In `dataset.csv`, they are 100% complete (0 missing).
- For `genre`: absent in `billboard_24years_lyrics_spotify.csv`; mapped directly to `track_genre` in `dataset.csv`.
- For `release year`: `year` in `billboard_24years_lyrics_spotify.csv` indicates the Billboard Hot 100 Year-End chart year (not necessarily original track release year); in `dataset.csv`, release year is absent (no date column; only album title string occasionally contains years).

### 3. Columns to treat as target leakage

**Identified Leakage Columns:**

1. **In `billboard_24years_lyrics_spotify.csv`:**
   - `ranking`: Direct post-release chart performance (Billboard Hot 100 year-end rank from 1 to 100). Pure leakage.
   - `year`: Chart year (identifies the year the song achieved peak year-end status on Billboard).
   - `songurl`, `url`: Wikipedia article URLs created specifically because the song/artist charted.
   - `lyrics`: May contain post-release Genius metadata annotations (e.g. scrape artifacts like "You might also like", "Embed").

2. **In `dataset.csv`:**
   - `popularity`: Spotify's 0-100 metric calculated dynamically from total streams and recent listening volume. Popularity is a post-release outcome of commercial success; using it to predict charting constitutes severe target leakage if charting is the target.
   - `Unnamed: 0`: Arbitrary sequential index column from dataset export (not leakage per se, but an uninformative artifact to drop).

### 4. What is the current class balance if we use your answer to Q1?

**Answer: UNCERTAIN**

Because there is no established binary target `charted` in either file, the class balance depends entirely on how `charted` is defined:
- **If evaluated solely inside `billboard_24years_lyrics_spotify.csv`:**
  - Class balance: **100.00% positive (3,397 / 3,397 rows are charted songs), 0.00% negative (0 rows)**. No negative class exists. (Or if restricted to rows with non-null audio features: **486 positive, 0 negative**).
- **If `dataset.csv` is mapped against `billboard_24years_lyrics_spotify.csv`:**
  - By exact Spotify track ID match: **56 charted (0.05%) vs 113,944 uncharted (99.95%)**.
  - By normalized track name + artist match: **462 unique matched tracks (~600–800 duplicate rows) (~0.5% positive vs ~99.5% negative)**.
- **If `popularity` in `dataset.csv` were used as a proxy threshold:**
  - Popularity >= 50: **29,367 (25.76%) positive vs 84,633 (74.24%) negative**.
  - Popularity >= 60: **14,822 (13.00%) positive vs 99,178 (87.00%) negative**.
  - Popularity >= 70: **5,472 (4.80%) positive vs 108,528 (95.20%) negative**.
  - Popularity >= 80: **1,201 (1.05%) positive vs 112,799 (98.95%) negative**.

Under Rule 6, the exact definition of `charted` must be resolved with the user before fixing class balance.
