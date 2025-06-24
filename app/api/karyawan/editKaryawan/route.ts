import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import bcrypt from "bcryptjs"; // ✅ Tambahkan ini

export async function PUT(req: NextRequest) {
  try {
    const body = await req.json();
    const { id, name, nip, email, password } = body;

    if (!id || !name?.trim() || !nip?.trim()) {
      return NextResponse.json(
        { message: "Nama, NIP, dan ID harus diisi" },
        { status: 400 }
      );
    }

    const updatedData: any = {
      nama: name.trim(),
      nip: nip.trim(),
    };

    if (email?.trim()) updatedData.email = email.trim();

    // ✅ Hash password jika ada
    if (password) {
      const hashedPassword = await bcrypt.hash(password, 10);
      updatedData.password = hashedPassword;
    }

    const updatedKaryawan = await prisma.karyawan.update({
      where: { id: Number(id) },
      data: updatedData,
    });

    return NextResponse.json({
      message: "Karyawan berhasil diperbarui",
      data: updatedKaryawan,
    });
  } catch (error: any) {
    console.error("PUT /api/karyawan error:", error);
    return NextResponse.json(
      { message: "Terjadi kesalahan saat memperbarui data karyawan" },
      { status: 500 }
    );
  }
}
