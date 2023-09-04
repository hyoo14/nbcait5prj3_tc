## Version
- `v1.1` : Baseline Model inference 결과와 predefined_news_category 값을 기반으로 1,371 개의 label noise 데이터를 필터링하였음.
	- 필터링 방식 : Model prediction 결과와 target 값이 다른 데이터에 대해서 prediction 결과가 predefined_news_category와 다른 데이터를 제거하였음.
	- 데이터 개수 변화 : 52530 -> 52374 (중복 데이터 제거) -> 51690 개 (684 개의 noise 데이터 제거)


