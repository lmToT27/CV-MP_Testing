# 🎵 Giả lập Sáo Mèo (Sáo H'Mông)

<p align="center">
  <a href="./README.md">English</a> &nbsp;|&nbsp; <b>Tiếng Việt</b>
</p>

---

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)](https://google.github.io/mediapipe/)

Tái tạo âm thanh da diết của **Sáo Mèo** dân tộc H'Mông thông qua cử chỉ bàn tay.

## 🏔 Về nhạc cụ Sáo Mèo
**Sáo Mèo** là loại nhạc cụ đặc trưng của người H'Mông ở vùng núi phía Bắc Việt Nam. Khác với sáo trúc thông thường, Sáo Mèo có một **lưỡi gà (lam đồng)** ở miệng thổi. Điều này tạo ra âm sắc rung, trầm ấm và da diết như tiếng người hát. Theo truyền thống, các chàng trai H'Mông dùng tiếng sáo này để tỏ tình ("gọi bạn") trong các đêm trăng.

Dự án này sử dụng **PyAudio** để tổng hợp âm thanh lưỡi gà và **MediaPipe** để chuyển đổi cách bấm ngón tay thành điều khiển kỹ thuật số.

## 🎮 Hướng dẫn chơi

### 🖐 Tay Trái: Điều khiển Quãng & Dấu Hóa
Tay trái đóng vai trò như việc điều tiết hơi và các lỗ bấm chuyển quãng.

| Bộ phận | Cử chỉ | Tác dụng |
| :--- | :--- | :--- |
| **Ngón cái** | **Mở** | Chơi nốt Giáng (**$\flat$**) |
| | **Gập** | Chơi nốt thường (Tự nhiên) |
| **Các ngón khác** | **0 ngón mở** | **Quãng 2** (Trầm) |
| | **1 ngón mở** | **Quãng 3** (Trung) |
| | **2 ngón mở** | **Quãng 4** (Cao) |
| | **...** | **...** |

### ✋ Tay Phải: Giai điệu (Nốt nhạc)
Tay phải điều khiển cao độ bằng cách chia một quãng tám thành 2 phần, sử dụng ngón cái để chuyển đổi.

#### 1. Ngón cái (Phím chuyển)
Ngón cái hoạt động như một "phím Shift" để chuyển giữa các nốt thấp và cao.
* **Gập:** Chơi **4 nốt đầu** của quãng (Đô, Rê, Mi, Fa).
* **Mở:** **Dịch lên một quãng 5** (Cộng thêm 3.5 cung) để chơi các nốt cao (Sol, La, Si...).

#### 2. Bảng ngón tay

| Số ngón mở | Ngón cái **GẬP** (Nốt thấp) | Ngón cái **MỞ** (Nốt cao) |
| :---: | :---: | :---: |
| **1** | **C** (Đô) | **G** (Sol) |
| **2** | **D** (Rê) | **A** (La) |
| **3** | **E** (Mi) | **B** (Si) |
| **4** | **F** (Fa) | **C** (Đô - Quãng tiếp) |

> **Cơ chế:** Với cách này, bạn có thể chơi trọn vẹn 7 nốt nhạc (Đồ -> Si) chỉ bằng việc kết hợp 4 ngón tay và đóng/mở ngón cái.

## 🛠 Cài đặt & Chạy chương trình

Bạn không cần cài đặt Python hay các thư viện phức tạp. Chỉ cần tải về và chạy file thực thi.

> **⚠️ Quan trọng:** Hãy đảm bảo máy tính của bạn đã cài **[Git LFS](https://git-lfs.github.com/)** để tải được file `.exe` (nếu không bạn sẽ chỉ thấy một file lỗi 1KB).

```bash
# 1. Tải bộ mã nguồn
git clone https://github.com/lmToT27/CV-MP_Testing.git

# 2. Truy cập thư mục
cd CV-MP_Testing

# 3. Chạy chương trình
# (Trên Windows)
.\dist\main.exe