"use client"

import { Users, UserPlus, LogOut, HomeIcon as HouseIcon, Settings, FileText, GraduationCap } from "lucide-react"
import { useRouter, usePathname } from "next/navigation"
import { logout } from "@/lib/auth"
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    useSidebar,
} from "@/components/ui/sidebar"

const navigation = [
    {
        name: "Dashboard",
        href: "/dashboard",
        icon: HouseIcon,
    },
    {
        name: "Manajemen",
        href: "/dashboard/management",
        icon: Users,
    },
    {
        name: "Pendaftaran",
        href: "/dashboard/registration",
        icon: UserPlus,
    },

]

export function AppSidebar() {
    const router = useRouter()
    const pathname = usePathname()
    const { close } = useSidebar()

    const handleLogout = async () => {
        if (confirm("Apakah Anda yakin ingin keluar?")) {
            try {
                const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL
                await fetch(`${BASE_URL}/api/logout`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                })
            } catch (error) {
                console.log("Logout API call failed, but continuing with local logout")
            }

            // Use the enhanced logout function from auth.ts
            logout()
        }
    }

    const handleNavigation = (href: string) => {
        router.push(href)
        close()
    }

    return (
        <Sidebar className="flex flex-col h-full bg-gradient-to-b from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-gray-800 border-r border-emerald-100 dark:border-gray-700">
            <SidebarHeader className="border-b border-emerald-100 dark:border-gray-700 bg-white/50 dark:bg-gray-800/50 p-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 lg:w-12 lg:h-12 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center shadow-lg flex-shrink-0">
                        <GraduationCap className="h-5 w-5 lg:h-7 lg:w-7 text-white" />
                    </div>
                    <div className="min-w-0 flex-1">
                        <h2 className="text-base lg:text-lg font-bold text-gray-900 dark:text-white truncate">Admin Dashboard</h2>
                        <p className="text-xs text-emerald-600 dark:text-emerald-400 font-medium truncate">Computer Vision</p>
                    </div>
                </div>
            </SidebarHeader>
        
            <SidebarContent className="bg-transparent flex-1 overflow-y-auto">
                <SidebarGroup>
                    <SidebarGroupLabel className="text-emerald-700 dark:text-emerald-300 font-semibold px-4 py-2">
                        Menu Utama
                    </SidebarGroupLabel>
                    <SidebarGroupContent className="px-2">
                        <SidebarMenu>
                            {navigation.map((item) => {
                                const isActive = pathname === item.href
                                return (
                                    <SidebarMenuItem key={item.name}>
                                        <SidebarMenuButton
                                            onClick={() => handleNavigation(item.href)}
                                            isActive={isActive}
                                            className={`w-full justify-start gap-3 transition-all duration-200 mx-2 my-1 ${isActive
                                                ? "bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-lg hover:from-emerald-600 hover:to-teal-700"
                                                : "text-gray-700 dark:text-gray-300 hover:bg-emerald-100 dark:hover:bg-emerald-900/20 hover:text-emerald-700 dark:hover:text-emerald-300"
                                                }`}
                                        >
                                            <item.icon className="h-4 w-4 flex-shrink-0" />
                                            <span className="font-medium truncate">{item.name}</span>
                                        </SidebarMenuButton>
                                    </SidebarMenuItem>
                                )
                            })}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>




            </SidebarContent>

            <SidebarFooter className="border-t border-emerald-100 dark:border-gray-700 bg-white/50 dark:bg-gray-800/50 p-4">
                <SidebarMenu>
                    <SidebarMenuItem>
                        <SidebarMenuButton
                            onClick={handleLogout}
                            className="w-full justify-start gap-3 text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-900/20 transition-all duration-200 font-medium"
                        >
                            <LogOut className="h-4 w-4 flex-shrink-0" />
                            <span className="truncate">Keluar</span>
                        </SidebarMenuButton>
                    </SidebarMenuItem>
                </SidebarMenu>
            </SidebarFooter>
        </Sidebar>
    )
}
