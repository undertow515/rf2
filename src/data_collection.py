"""
KMA API를 통한 시간 단위 강수 데이터 수집
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os


class KMAHourlyDataCollector:
    """기상청 API를 통해 시간 단위 강수 데이터를 수집하는 클래스"""
    
    def __init__(self, auth_key, stn_id, start_date, end_date, output_dir='./data'):
        """
        Parameters
        ----------
        auth_key : str
            기상청 API 인증키
        stn_id : int
            관측소 ID (예: 108=서울, 159=부산)
        start_date : datetime
            시작 날짜
        end_date : datetime
            종료 날짜
        output_dir : str
            데이터 저장 디렉토리
        """
        self.auth_key = auth_key
        self.stn_id = stn_id
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.base_url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
        
        os.makedirs(output_dir, exist_ok=True)
        
    def parse_line(self, line):
        """한 줄의 데이터를 파싱"""
        if line.startswith('#') or len(line) < 100:
            return None
            
        try:
            time_str = line[0:12]
            dt = datetime.strptime(time_str, '%Y%m%d%H%M')
            stn = line[12:17].strip()
            rn_str = line[93:99].strip()
            
            if rn_str == '-9.0' or rn_str == '-9':
                precip = None
            else:
                precip = float(rn_str)
                
            return {
                'datetime': dt,
                'stn_id': stn,
                'precip_mm': precip
            }
        except Exception:
            return None
    
    def fetch_data_for_datetime(self, dt):
        """특정 시각의 데이터를 API로부터 가져오기"""
        tm = dt.strftime('%Y%m%d%H%M')
        
        params = {
            'tm': tm,
            'stn': self.stn_id,
            'help': 0,
            'authKey': self.auth_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                lines = response.text.splitlines()
                for line in lines:
                    parsed = self.parse_line(line)
                    if parsed and parsed['stn_id'] == str(self.stn_id):
                        return parsed
            return None
        except Exception as e:
            print(f"Error fetching {tm}: {e}")
            return None
    
    def collect_all_data(self, save_interval=24*30):
        """전체 기간의 데이터를 수집"""
        print(f"데이터 수집 시작: {self.start_date} ~ {self.end_date}")
        print(f"관측소 ID: {self.stn_id}")
        
        current_dt = self.start_date
        data_list = []
        count = 0
        
        total_hours = int((self.end_date - self.start_date).total_seconds() / 3600) + 1
        pbar = tqdm(total=total_hours, desc='Data Collection')
        
        while current_dt <= self.end_date:
            data = self.fetch_data_for_datetime(current_dt)
            
            if data:
                data_list.append(data)
            
            count += 1
            
            if count % save_interval == 0:
                self._save_intermediate(data_list)
                data_list = []
            
            time.sleep(0.1)  # API 과부하 방지
            
            current_dt += timedelta(hours=1)
            pbar.update(1)
        
        pbar.close()
        
        if data_list:
            self._save_intermediate(data_list)
        
        self._merge_all_files()
        print(f"\n✓ 데이터 수집 완료!")
        
    def _save_intermediate(self, data_list):
        """중간 데이터 저장"""
        if not data_list:
            return
            
        df = pd.DataFrame(data_list)
        df = df.sort_values('datetime')
        
        filename = f'intermediate_{self.stn_id}_{df["datetime"].iloc[0].strftime("%Y%m%d")}.csv'
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath, index=False)
    
    def _merge_all_files(self):
        """모든 중간 파일을 하나로 병합"""
        print("\n중간 파일 병합 중...")
        
        intermediate_files = [f for f in os.listdir(self.output_dir) 
                             if f.startswith(f'intermediate_{self.stn_id}_')]
        
        if not intermediate_files:
            return
        
        df_list = []
        for filename in sorted(intermediate_files):
            filepath = os.path.join(self.output_dir, filename)
            df_temp = pd.read_csv(filepath, parse_dates=['datetime'])
            df_list.append(df_temp)
        
        df_final = pd.concat(df_list, ignore_index=True)
        df_final = df_final.sort_values('datetime')
        df_final = df_final.drop_duplicates(subset=['datetime'])
        
        station_names = {
            108: 'seoul', 159: 'busan', 105: 'gangneung', 
            127: 'chungju', 136: 'andong', 156: 'gwangju',
            168: 'yeosu', 184: 'jeju'
        }
        station_name = station_names.get(int(self.stn_id), f'stn{self.stn_id}')
        
        final_filename = f'{station_name}_hourly.csv'
        final_filepath = os.path.join(self.output_dir, final_filename)
        
        df_final.to_csv(final_filepath, index=False)
        print(f"✓ 최종 파일 저장: {final_filename}")
        print(f"  - 총 레코드 수: {len(df_final)}")
        
        # 중간 파일 삭제
        for filename in intermediate_files:
            filepath = os.path.join(self.output_dir, filename)
            os.remove(filepath)


# 관측소 정보
STATION_INFO = {
    'seoul': 108,
    'busan': 159,
    'gangneung': 105,
    'chungju': 127,
    'andong': 136,
    'gwangju': 156,
    'yeosu': 168,
    'jeju': 184
}