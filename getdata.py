import csv
from googleapiclient.discovery import build

# ================= CẤU HÌNH =================
DEVELOPER_KEY = "AIzaSyBosShK5z9bqY6OUzmbHZRbxlHSeKYvha4"

VIDEO_ID = "nZtFlrwCbs4" 


CSV_FILE_NAME = "comments.csv"
# ============================================

def get_all_comments(api_key, video_id):
    # Khởi tạo client
    youtube = build('youtube', 'v3', developerKey=api_key)

    print(f"Đang lấy toàn bộ comment từ video: {video_id}...")
    
    # Tạo list chứa kết quả
    comments_data = []

    # Tạo request lấy trang đầu tiên
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100, # Lấy tối đa mỗi lần gọi (max 100)
        textFormat="plainText"
    )

    try:
        # Vòng lặp lấy dữ liệu cho đến khi hết các trang (Pagination)
        while request:
            response = request.execute()

            for item in response['items']:
                # Truy cập vào cấu trúc dữ liệu của YouTube
                comment_info = item['snippet']['topLevelComment']['snippet']
                
                # Chỉ lấy 2 trường thông tin bạn cần
                user_name = comment_info['authorDisplayName']
                content = comment_info['textDisplay']

                comments_data.append({
                    "User": user_name,
                    "Text": content
                })

            # Kiểm tra xem còn trang sau không, nếu có thì gọi tiếp
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=response['nextPageToken']
                )
            else:
                break
                
        return comments_data

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        return []

def save_to_csv(data, filename):
    if not data:
        print("Không có dữ liệu để lưu.")
        return

    print(f"Đang lưu {len(data)} dòng vào file {filename}...")
    
    # Mở file CSV để ghi (encoding='utf-8-sig' để mở trên Excel không bị lỗi font tiếng Việt)
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=["User", "Text"])
        
        # Viết dòng tiêu đề (Header)
        writer.writeheader()
        
        # Viết nội dung
        writer.writerows(data)
    
    print("Xong! Kiểm tra file CSV của bạn.")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # 1. Gọi hàm lấy dữ liệu
    result = get_all_comments(DEVELOPER_KEY, VIDEO_ID)
    
    # 2. Gọi hàm lưu file
    if result:
        save_to_csv(result, CSV_FILE_NAME)