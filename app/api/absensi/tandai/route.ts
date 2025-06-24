import { NextResponse } from "next/server"
import prisma from "@/lib/prisma"
import { AttendanceResponse } from "@/lib/api"
import dayjs from 'dayjs'
import utc from 'dayjs/plugin/utc'
import timezone from 'dayjs/plugin/timezone'

dayjs.extend(utc)
dayjs.extend(timezone)

const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL
const FLASK_ATTENDANCE_URL = `${BASE_URL}/attendance`
const OFFICE_LAT = parseFloat(process.env.NEXT_PUBLIC_OFFICE_LAT || "0");
const OFFICE_LNG = parseFloat(process.env.NEXT_PUBLIC_OFFICE_LNG || "0");
const MAX_DISTANCE_METERS = parseInt(process.env.NEXT_PUBLIC_MAX_DISTANCE_METERS || "1200000000");

function calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371000
  const dLat = (lat2 - lat1) * (Math.PI / 180)
  const dLon = (lon2 - lon1) * (Math.PI / 180)
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * (Math.PI / 180)) *
    Math.cos(lat2 * (Math.PI / 180)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return R * c
}

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { image, latitude, longitude } = body

    if (!image || typeof latitude !== 'number' || typeof longitude !== 'number') {
      return NextResponse.json({ message: 'Data tidak lengkap: foto dan lokasi wajib diisi' }, { status: 400 })
    }

    const distance = calculateDistance(latitude, longitude, OFFICE_LAT, OFFICE_LNG)
    if (distance > MAX_DISTANCE_METERS) {
      return NextResponse.json({
        message: `Lokasi di luar jangkauan absensi (${Math.round(distance)} meter dari kantor)`,
      }, { status: 403 })
    }

    // Kirim ke Flask
    const flaskRes = await fetch(FLASK_ATTENDANCE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    })

    const flaskData = await flaskRes.json()

    if (!flaskRes.ok || !flaskData.nip) {
      return NextResponse.json({
        success: false,
        message: flaskData.message || flaskData.error || 'Wajah tidak dikenali.',
      }, { status: flaskRes.status === 200 ? 404 : flaskRes.status })
    }

    const { nip } = flaskData
    const karyawan = await prisma.karyawan.findUnique({ where: { nip: String(nip) } })

    if (!karyawan) {
      return NextResponse.json({
        success: false,
        message: `Karyawan dengan NIP ${nip} tidak ditemukan.`,
      }, { status: 404 })
    }

    const pengaturan = await prisma.pengaturanAbsensi.findFirst()
    if (!pengaturan) {
      return NextResponse.json({
        success: false,
        message: 'Pengaturan absensi tidak ditemukan.',
      }, { status: 500 })
    }

      // Gunakan waktu saat ini di Asia/Jakarta (UTC+7) untuk logika
   const nowWIB = dayjs().tz('Asia/Jakarta')

  // Tetap simpan ke database dalam bentuk Date JS biasa
   const now = nowWIB.toDate()



    const awalHari = nowWIB.startOf('day').toDate()
    const akhirHari = nowWIB.endOf('day').toDate()


    // 🔒 Cek Hari Libur Nasional
    const liburHariIni = await prisma.hariLibur.findFirst({
      where: {
        tanggal: {
          gte: awalHari,
          lte: akhirHari,
        },
      },
    })

    if (liburHariIni) {
      return NextResponse.json({
        success: false,
        message: `Hari ini adalah hari libur: ${liburHariIni.keterangan}. Absensi tidak diperbolehkan.`,
      }, { status: 403 })
    }

    // 🔒 Cek Hari Kerja
    const hariInggris = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
    const hariIni = hariInggris[now.getDay()]
    if (!pengaturan.hariKerja.includes(hariIni)) {
      return NextResponse.json({
        success: false,
        message: `Hari ${hariIni} bukan hari kerja.`,
      }, { status: 403 })
    }

    // 🔍 Cek absensi hari ini
    const semuaAbsensiHariIni = await prisma.catatanAbsensi.findMany({
      where: {
        karyawanId: karyawan.id,
        timestamp_absensi: { gte: awalHari, lte: akhirHari },
      },
    })

    if (semuaAbsensiHariIni.length >= 2) {
      return NextResponse.json({
        success: false,
        message: 'Karyawan Sudah Melakukan Absen Masuk dan Pulang.',
      }, { status: 403 })
    }

    const formatJam = (jamStr: string) => {
  const [h, m] = jamStr.split(':').map(Number)
  return nowWIB.clone().hour(h).minute(m).second(0).millisecond(0).toDate()
}


    const waktuMulaiMasuk = formatJam(pengaturan.waktuMulaiAbsen)
    const batasMasuk = formatJam(pengaturan.batasTerlambat)
    const waktuMulaiPulang = formatJam(pengaturan.waktuMulaiPulang)
    const batasPulang = formatJam(pengaturan.batasWaktuPulang)
    console.log(now)
    console.log(waktuMulaiMasuk)
    console.log(batasMasuk)
    console.log(waktuMulaiPulang)

    let statusAbsensi: string

    if (now >= waktuMulaiMasuk && now <= batasMasuk) {
      const sudahMasuk = semuaAbsensiHariIni.some(a =>
        a.status === 'tepat waktu' || a.status === 'terlambat'
      )

      if (sudahMasuk) {
        return NextResponse.json({
          success: false,
          message: 'Karyawan sudah melakukan absensi masuk hari ini.',
        }, { status: 403 })
      }

      statusAbsensi = now <= formatJam(pengaturan.batasTepatWaktu)
        ? 'tepat waktu'
        : 'terlambat'

    } else if (now >= waktuMulaiPulang && now <= batasPulang) {
      const sudahPulang = semuaAbsensiHariIni.some(a => a.status === 'pulang')

      if (sudahPulang) {
        return NextResponse.json({
          success: false,
          message: 'Karyawan sudah melakukan absensi pulang hari ini.',
        }, { status: 403 })
      }

      statusAbsensi = 'pulang'

    } else {
      return NextResponse.json({
        success: false,
        message: 'Sekarang bukan waktu absensi yang diperbolehkan.',
      }, { status: 403 })
    }

    const newAttendanceRecord = await prisma.catatanAbsensi.create({
      data: {
        karyawanId: karyawan.id,
        timestamp_absensi: now,
        status: statusAbsensi,
      },
    })

    const responsePayload: AttendanceResponse = {
      success: true,
      message: `Absensi ${statusAbsensi} untuk ${karyawan.nama} berhasil dicatat!`,
      nama: karyawan.nama,
      nip: karyawan.nip,
      timestamp: now.toISOString(),
      status: statusAbsensi,
      catatanId: newAttendanceRecord.id,
    }

    return NextResponse.json(responsePayload, { status: 200 })
  } catch (error: any) {
    console.error('Error di API /api/absensi/tandai:', error)
    return NextResponse.json({
      success: false,
      message: 'Terjadi kesalahan pada server.',
      errorDetail: error.message,
    }, { status: 500 })
  }
}
