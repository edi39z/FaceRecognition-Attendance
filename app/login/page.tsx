"use client"

import type React from "react"

import { useRouter } from "next/navigation"
import { useState, useEffect } from "react"
import { Eye, EyeOff, Mail, Lock, User, GraduationCap } from "lucide-react"
import { setUserRole, isAuthenticated } from "@/lib/auth"

export default function LoginPage() {
    const router = useRouter()

    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    const [error, setError] = useState("")
    const [loading, setLoading] = useState(false)
    const [showPassword, setShowPassword] = useState(false)

    // Check if user is already authenticated
    useEffect(() => {
        if (isAuthenticated()) {
            const userRole = localStorage.getItem("role")
            if (userRole === "admin") {
                router.replace("/dashboard")
            } else if (userRole === "user") {
                router.replace("/")
            }
        }
    }, [router])

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault()
        setError("")
        setLoading(true)

        if (!email || !password) {
            setError("Email dan password wajib diisi.")
            setLoading(false)
            return
        }

        try {
            const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL
            const res = await fetch(`${BASE_URL}/api/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({email,password }),
            })

            const data = await res.json()

            if (res.ok) {
                // Simpan data user berdasarkan role menggunakan fungsi auth yang baru
                setUserRole(data.role as "admin" | "user")

                // Simpan data tambahan ke localStorage untuk backward compatibility
                if (data.role === "admin") {
                    localStorage.setItem("userType", "admin")
                    router.replace("/dashboard")
                } else if (data.role === "user") {
                    localStorage.setItem("nama", data.nama)
                    localStorage.setItem("userType", "karyawan")
                    router.replace("/")
                }
            } else {
                setError(data.message || data.error || "Login gagal")
            }
        } catch (err) {
            setError("Gagal terhubung ke server Flask.")
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-100 flex items-center justify-center p-4">
            <div className="w-full max-w-4xl">
                <div className="bg-white rounded-3xl shadow-2xl overflow-hidden flex">
                    <div className="w-full lg:w-1/2 p-8 lg:p-12">
                        <div className="max-w-md mx-auto">
                            <div className="text-center mb-8">
                                <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                    <User className="h-8 w-8 text-emerald-600" />
                                </div>
                                <h1 className="text-2xl font-bold text-gray-800 mb-2">Selamat Datang</h1>
                                <p className="text-gray-500 text-sm">Silakan masuk ke akun Anda</p>
                            </div>

                            <form onSubmit={handleLogin} className="space-y-6">
                                {/* Email */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                                    <div className="relative">
                                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                            <Mail className="h-5 w-5 text-gray-400" />
                                        </div>
                                        <input
                                            type="email"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl bg-gray-50 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                                            placeholder="Masukkan email Anda"
                                            required
                                        />
                                    </div>
                                </div>

                                {/* Password  */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                                    <div className="relative">
                                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                            <Lock className="h-5 w-5 text-gray-400" />
                                        </div>
                                        <input
                                            type={showPassword ? "text" : "password"}
                                            value={password}
                                            onChange={(e) => setPassword(e.target.value)}
                                            className="w-full pl-10 pr-12 py-3 border border-gray-200 rounded-xl bg-gray-50 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                                            placeholder="Masukkan password Anda"
                                            required
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowPassword(!showPassword)}
                                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 transition-colors"
                                        >
                                            {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                                        </button>
                                    </div>
                                </div>



                                {/* Error Message */}
                                {error && (
                                    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm">
                                        {error}
                                    </div>
                                )}

                                {/* Login Button */}
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded-xl transition-all duration-200 disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-[0.98]"
                                >
                                    {loading ? (
                                        <div className="flex items-center justify-center">
                                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                                            Memproses...
                                        </div>
                                    ) : (
                                        "Masuk"
                                    )}
                                </button>
                            </form>
                        </div>
                    </div>

                    {/* block samping kanan */}
                    <div className="hidden lg:block lg:w-1/2 relative bg-gradient-to-br from-emerald-600 to-teal-700 overflow-hidden">
                        <div className="absolute inset-0 opacity-5">
                            <div className="absolute top-0 left-0 w-full h-full">
                                <div className="grid grid-cols-6 gap-8 h-full p-8 transform rotate-45 scale-150">
                                    {Array.from({ length: 36 }).map((_, i) => (
                                        <div
                                            key={i}
                                            className="bg-white rounded-lg"
                                            style={{
                                                animationDelay: `${i * 0.2}s`,
                                                animation: "pulse 4s infinite",
                                            }}
                                        />
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="relative z-10 h-full flex flex-col justify-center items-center p-12 text-white">
                            <div className="text-center mb-12">
                                <div className="w-20 h-20 bg-white/15 backdrop-blur-sm rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-lg">
                                    <GraduationCap className="h-10 w-10 text-white" />
                                </div>

                                <div className="space-y-3">
                                    <h1 className="text-3xl font-bold leading-tight">Computer Vision</h1>
                                    <h2 className="text-4xl font-extrabold bg-gradient-to-r from-white to-emerald-100 bg-clip-text text-transparent">
                                      Universitas Sumatera Utara
                                    </h2>
                                    <h3 className="text-2xl font-semibold text-emerald-100">Medan</h3>
                                </div>

                                <div className="mt-8 space-y-2">
                                    <p className="text-emerald-100 text-lg font-medium">Sistem Absensi Digital</p>
                                    <div className="w-24 h-1 bg-gradient-to-r from-emerald-300 to-teal-300 rounded-full mx-auto"></div>
                                </div>
                            </div>


                            <div className="absolute top-8 right-8 w-16 h-16 border-2 border-white/20 rounded-lg transform rotate-45"></div>
                            <div className="absolute bottom-8 left-8 w-12 h-12 border-2 border-white/20 rounded-lg transform rotate-45"></div>
                            <div className="absolute top-1/4 left-12 w-8 h-8 border-2 border-white/20 rounded-lg transform rotate-45"></div>
                        </div>

                        {/* Bottom Decorative Wave */}
                        <div className="absolute bottom-0 left-0 w-full">
                            <svg viewBox="0 0 1200 120" preserveAspectRatio="none" className="relative block w-full h-12">
                                <path
                                    d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z"
                                    className="fill-white opacity-10"
                                />
                            </svg>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
