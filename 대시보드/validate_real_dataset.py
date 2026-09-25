"""
새 실데이터 CSV(아파트_거래_DATA_대시보드_실데이터.csv)를 APARTMENT_DATA_CSV로 지정했을 때
스트림릿 앱이 정상 동작하는지 확인하는 임시 검증 스크립트 (기존 test_dashboard.py와 별개).

실행: python validate_real_dataset.py
"""
import os
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from streamlit.testing.v1 import AppTest

REAL_CSV = Path(os.environ.get(
    "VALIDATE_CSV",
    str(APP_DIR.parent / "거래_단지_DATA" / "아파트_거래_DATA_대시보드_실데이터.csv"),
))


def main():
    assert REAL_CSV.exists(), f"파일이 없습니다: {REAL_CSV}"
    os.environ["APARTMENT_DATA_CSV"] = str(REAL_CSV)

    app = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=120).run()
    print("초기 로딩 예외:", app.exception)
    assert not app.exception

    district_options = app.selectbox(key="real_district").options
    print(f"자치구 옵션 수: {len(district_options)}")
    print("자치구 옵션 샘플:", district_options[:5])
    assert len(district_options) > 20  # 서울 25개 구 전체가 있어야 함

    complex_options = app.selectbox(key="real_complex").options
    print(f"현재 자치구 내 단지 수: {len(complex_options)}")
    print("현재 선택된 단지:", app.selectbox(key="real_complex").value)

    # 다른 자치구로 변경 후 결과 보기
    app.selectbox(key="real_district").select("강남구").run()
    app.button[0].click().run()
    print("강남구 적용 후 예외:", app.exception)
    assert not app.exception
    print("강남구 적용된 단지:", app.session_state["domain_real"]["applied"])

    # 시연 화면 모드 전환도 여전히 동작하는지 확인
    app.radio(key="view_mode").set_value("시연 화면").run()
    print("시연 화면 전환 예외:", app.exception)
    assert not app.exception
    app.radio(key="view_mode").set_value("실데이터").run()
    assert not app.exception

    all_markdown = "\n".join(el.value for el in app.markdown)
    assert "실거래 CSV 기준" in all_markdown
    print("소스 스트립에 '실거래 CSV 기준' 문구 확인됨")

    print("\n모든 검증 통과.")


if __name__ == "__main__":
    main()
